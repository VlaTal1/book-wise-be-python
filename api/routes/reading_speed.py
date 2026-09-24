import logging
import uuid

import jwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status as http_status

from core.security import verify_token
from models.reading_speed import ReadingSpeedResult
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
    text_id: str = Query(default="text_1"),
):
    try:
        user_id = verify_token(token)
    except jwt.PyJWTError as e:
        logger.warning(f"WS auth failed: {e}")
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return

    try:
        reference = load_reference_text(text_id)
    except FileNotFoundError:
        await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION, reason=f"Unknown text_id: {text_id}")
        return

    await websocket.accept()
    session = ReadingSpeedSession(session_id=str(uuid.uuid4()), user_id=user_id, reference=reference)
    logger.info(
        f"Reading-speed session {session.session_id} started for user {user_id}, "
        f"participant {participant_id}, text {text_id}"
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
        return

    events, metrics, all_word_events = session.finalize()
    for event in events:
        await websocket.send_json(event.model_dump())

    # Зберігаємо в Java СИНХРОННО (до відправки "result" клієнту), щоб мати
    # attempt_id — мобілка використає його для опитування стану окремого
    # шару перевірки наголосу (рахується в фоні, після закриття сокета,
    # див. services/stress_background.py). participant_id довіряємо як є
    # (мобілка вже пройшла Supabase JWT) — без перевірки, що цей participant
    # дійсно належить user_id; Java-ендпоінт /internal/** теж не має
    # контексту користувача для такої перевірки. Прийнятний компроміс для
    # MVP; якщо знадобиться суворіша ізоляція — Python має звертатися до
    # Java за належністю participant->user.
    attempt_id = report_reading_speed_attempt(participant_id, reference, metrics, all_word_events)

    result = ReadingSpeedResult(text_id=reference.id, metrics=metrics, attempt_id=attempt_id)
    await websocket.send_json(result.model_dump())
    await websocket.close()
    logger.info(f"Reading-speed session {session.session_id} finished: {metrics}")

    if attempt_id is not None:
        schedule_stress_check(
            attempt_id=attempt_id,
            session_id=session.session_id,
            audio_path=session.audio_path(),
            reference_words=[w.word for w in reference.words],
        )
