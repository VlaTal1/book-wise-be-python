import requests

from core.config import settings

# REST-виклик Gemini API напряму через requests: офіційний SDK google-genai
# потребує pydantic>=2.12, а проєкт зафіксований на pydantic 2.3.
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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
    parts = response.json()["candidates"][0]["content"]["parts"]
    # частини з міркуваннями моделі (thought) не є відповіддю
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))
