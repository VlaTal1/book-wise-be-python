import requests

from core.config import settings

# REST-виклик Gemini API напряму через requests: офіційний SDK google-genai
# потребує pydantic>=2.12, а проєкт зафіксований на pydantic 2.3.
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Причини, з якими Gemini детерміновано відмовляє на цей самий текст —
# повторювати такий запит марно.
BLOCK_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class ContentBlockedError(RuntimeError):
    """Gemini відмовився відповідати через вміст запиту (фільтр безпеки)."""


def generate_answer(prompt: str) -> str:
    """Текстова відповідь моделі (settings.llm_model) на запит. Температура не
    задається: для моделей Gemini 3 рекомендовано типове значення, і саме з ним
    модель обирали в master/quiz_exp."""
    response = requests.post(
        GEMINI_URL.format(model=settings.llm_model),
        headers={"x-goog-api-key": settings.gemini_api_key},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()

    # Gemini може відповісти 200 без кандидатів (запит заблоковано фільтром
    # безпеки) або з кандидатом без тексту — даємо зрозумілу помилку з причиною
    # замість KeyError: 'candidates'.
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise ContentBlockedError(f"Gemini blocked the prompt: {reason}")
        raise RuntimeError("Gemini returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts")
    if not parts:
        finish_reason = candidates[0].get("finishReason", "unknown")
        if finish_reason in BLOCK_FINISH_REASONS:
            raise ContentBlockedError(f"Gemini blocked the answer: {finish_reason}")
        raise RuntimeError(f"Gemini returned empty answer: {finish_reason}")
    # частини з міркуваннями моделі (thought) не є відповіддю
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))
