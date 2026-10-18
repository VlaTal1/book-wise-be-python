import asyncio
import logging
import uuid

import jwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status as http_status

from core.security import verify_token
from models.reading_speed import ReadingSpeedResult
from services.book_excerpt import NoExcerptAvailableError, select_random_excerpt
from services.comprehension_quiz import generate_comprehension_quiz
from services.java_client import report_reading_speed_attempt
from services.reading_speed_session import ReadingSpeedSession
from services.stress_background import schedule_stress_check
from utils.reference_text import load_reference_text

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/reading-speed")
async def reading_speed_ws(
    websocket: WebSocket,
    token: str = Query(...),
    participant_id: int = Query(...),
    # Порожньо (за замовчуванням) -> випадковий уривок ~300-340 слів з книги,
    # призначеної дитині (services/book_excerpt.py). Явний text_id лишається
    # для розробки/діагностики (фіксований resources/reference_texts/*.json).
    text_id: str | None = Query(default=None),
):
    try:
        user_id = verify_token(token)
    except jwt.PyJWTError as e:
        logger.warning(f"WS auth failed: {e}")
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return

    try:
        if text_id:
            reference = load_reference_text(text_id)
        else:
            reference = await asyncio.to_thread(select_random_excerpt, participant_id)
    except FileNotFoundError:
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION, reason=f"Unknown text_id: {text_id}")
        return
    except NoExcerptAvailableError as e:
        logger.warning(f"No excerpt available for participant {participant_id}: {e}")
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION, reason="No book excerpt available")
        return

    await websocket.accept()
    session = ReadingSpeedSession(session_id=str(uuid.uuid4()), user_id=user_id, reference=reference)
    logger.info(
        f"Reading-speed session {session.session_id} started for user {user_id}, "
        f"participant {participant_id}, text {reference.id}"
    )

    await websocket.send_json({
        "type": "reference",
        "text_id": reference.id,
        "title": reference.title,
        "words": [w.model_dump() for w in reference.words],
    })

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                for event in session.process_audio_chunk(message["bytes"]):
                    await websocket.send_json(event.model_dump())
                if session.is_reading_complete():
                    # Розпізнавання дійшло до кінця тексту — не чекаємо
                    # client-side "stop": останні слова в буфері (TAIL_HOLDBACK)
                    # інакше ніколи самі не підтвердяться, бо після кінця
                    # тексту вже нема наступного контексту для коміту.
                    break
            elif message.get("text") == "stop":
                break
    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id} disconnected mid-stream")
        session.close()
        return

    events, metrics, all_word_events = session.finalize()
    for event in events:
        await websocket.send_json(event.model_dump())

    # Тест на розуміння залежить лише від прочитаного тексту, тому генерується
    # паралельно з перевіркою замін, а не після неї.
    quiz_task = asyncio.create_task(asyncio.to_thread(generate_comprehension_quiz, session.attempted_text()))

    # Гібридна перевірка замін (services/substitution_check.py): граматика не
    # бачить іншої форми слова, тому слова, які обидва розпізнавачі без
    # граматики почули інакше, перепозначаються як помилка ще ДО збереження
    # результату — метрики в Java і в "result" уже враховують ці помилки.
    try:
        changed, metrics, all_word_events = await asyncio.to_thread(session.run_substitution_check)
        for event in changed:
            await websocket.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Substitution check failed for session {session.session_id}: {e}")

    # Зберігаємо в Java СИНХРОННО (до відправки "result" клієнту), щоб мати
    # attempt_id — мобілка використає його для опитування стану окремого
    # шару перевірки наголосу (рахується в фоні, після закриття сокета,
    # див. services/stress_background.py). participant_id довіряємо як є
    # (мобілка вже пройшла Supabase JWT) — без перевірки, що цей participant
    # дійсно належить user_id; Java-ендпоінт /internal/** теж не має
    # контексту користувача для такої перевірки. Прийнятний компроміс для
    # MVP; якщо знадобиться суворіша ізоляція — Python має звертатися до
    # Java за належністю participant->user.
    attempt_id = report_reading_speed_attempt(
        participant_id, reference, metrics, all_word_events, session.audio_object_key()
    )

    # Той самий виклик Claude, що й у "великому" генераторі тестів
    # (services/test_generator.py), але з промптом під короткий уривок і без
    # розбиття на частини — секунди, а не хвилини, тому результат віддаємо
    # разом з "result", а не окремим повідомленням.
    try:
        quiz = await quiz_task
    except Exception as e:
        logger.error(f"Failed to generate comprehension quiz for session {session.session_id}: {e}")
        quiz = []

    result = ReadingSpeedResult(
        text_id=reference.id, metrics=metrics, attempt_id=attempt_id, quiz=quiz or None
    )
    await websocket.send_json(result.model_dump())
    await websocket.close()
    logger.info(f"Reading-speed session {session.session_id} finished: {metrics}")

    if attempt_id is not None:
        schedule_stress_check(
            attempt_id=attempt_id,
            session_id=session.session_id,
            audio_path=session.audio_path(),
            # Лише слова, до яких дитина реально дочитала — не весь уривок
            # (300-340 слів "з запасом"), інакше транскрипт для forced
            # alignment не відповідає ~60-секундному запису (NoAlignmentsError).
            reference_words=session.attempted_reference_words(),
        )
