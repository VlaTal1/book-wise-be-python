import json
import logging

import requests

from core.config import settings
from models.reading_speed import ReadingSpeedMetrics, ReferenceText, WordEvent

logger = logging.getLogger(__name__)


def report_reading_speed_attempt(
    participant_id: int,
    reference: ReferenceText,
    metrics: ReadingSpeedMetrics,
    word_events: list[WordEvent],
) -> None:
    """Надсилає результат сесії читання в Java (service-to-service, той самий
    внутрішній API-ключ з обох боків — рішення по п.2 в docs/reading-speed-
    feature-design.md, розділ 8). Не піднімає виняток при невдачі: результат
    вже показаний користувачу в WS, помилка збереження історії не повинна
    ламати сесію — лише логуємо."""
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
        logger.info(f"Reading-speed attempt saved in Java: participant={participant_id}, text={reference.id}")
    except requests.RequestException as e:
        logger.error(f"Failed to report reading-speed attempt to Java: {e}")
