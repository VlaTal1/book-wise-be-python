import json
import logging

import requests

from core.config import settings
from models.reading_speed import ReadingSpeedMetrics, ReferenceText, WordEvent
from models.stress import StressCheckResult

logger = logging.getLogger(__name__)


def report_reading_speed_attempt(
    participant_id: int,
    reference: ReferenceText,
    metrics: ReadingSpeedMetrics,
    word_events: list[WordEvent],
) -> int | None:
    """Надсилає результат сесії читання в Java (service-to-service, той самий
    внутрішній API-ключ з обох боків — рішення по п.2 в docs/reading-speed-
    feature-design.md, розділ 8). Повертає id збереженого attempt (потрібен,
    щоб пізніше окремо доповісти результат шару перевірки наголосу — див.
    services/stress_background.py) або None, якщо збереження не вдалося:
    результат уже показаний користувачу в WS, помилка збереження історії не
    повинна ламати сесію — лише логуємо, і шар наголосу в цьому разі просто
    не запускається (нема attempt id, до якого прив'язати результат)."""
    payload = {
        "participantId": participant_id,
        "textId": reference.id,
        "totalWords": metrics.total_words,
        "correctCount": metrics.correct_count,
        "errorCount": metrics.error_count,
        "skippedCount": metrics.skipped_count,
        "notInVocabularyCount": metrics.not_in_vocabulary_count,
        "durationSeconds": metrics.duration_seconds,
        "wpm": metrics.wpm,
        "accuracy": metrics.accuracy,
        "wordsJson": json.dumps([e.model_dump(mode="json") for e in word_events], ensure_ascii=False),
    }

    try:
        response = requests.post(
            f"{settings.java_base_url}/internal/reading-speed-attempts",
            json=payload,
            headers={"X-Internal-Api-Key": settings.java_internal_api_key},
            timeout=10,
        )
        response.raise_for_status()
        attempt_id = response.json().get("id")
        logger.info(f"Reading-speed attempt saved in Java: id={attempt_id}, participant={participant_id}, text={reference.id}")
        return attempt_id
    except requests.RequestException as e:
        logger.error(f"Failed to report reading-speed attempt to Java: {e}")
        return None


def report_stress_progress(attempt_id: int, status: str, progress: int) -> None:
    """Проміжне оновлення прогресу шару перевірки наголосу (не критичне —
    якщо не дійшло, мобілка просто побачить лоадер трохи довше, фінальний
    результат все одно прийде окремим викликом report_stress_result)."""
    try:
        response = requests.patch(
            f"{settings.java_base_url}/internal/reading-speed-attempts/{attempt_id}/stress-progress",
            json={"status": status, "progress": progress},
            headers={"X-Internal-Api-Key": settings.java_internal_api_key},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to report stress progress for attempt {attempt_id}: {e}")


def report_stress_result(
    attempt_id: int,
    status: str,
    result: StressCheckResult | None = None,
    error: str | None = None,
) -> None:
    """Фінальний результат шару перевірки наголосу (успіх або помилка)."""
    payload = {
        "status": status.upper(),
        "accuracy": result.accuracy if result else None,
        "checkedWords": result.checked_words if result else None,
        "correctWords": result.correct_words if result else None,
        "wordsJson": json.dumps([w.model_dump(mode="json") for w in result.words], ensure_ascii=False) if result else None,
        "error": error,
    }
    try:
        response = requests.patch(
            f"{settings.java_base_url}/internal/reading-speed-attempts/{attempt_id}/stress-result",
            json=payload,
            headers={"X-Internal-Api-Key": settings.java_internal_api_key},
            timeout=10,
        )
        response.raise_for_status()
        logger.info(f"Stress-check result saved in Java for attempt {attempt_id}: status={status}")
    except requests.RequestException as e:
        logger.error(f"Failed to report stress-check result for attempt {attempt_id}: {e}")
