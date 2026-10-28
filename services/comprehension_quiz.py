import logging
from typing import List

from pydantic import ValidationError

from models.reading_speed import ComprehensionQuestion
from services.llm_service import generate_answer
from utils import get_json_from_response
from utils.prompt_utils import get_comprehension_quiz_prompt

logger = logging.getLogger(__name__)

# Нижче цього майже нема що питати (AI і сам здатен визнати текст "занадто
# коротким", але для зовсім крихітних уривків не варто навіть витрачати виклик
# моделі).
MIN_WORDS_FOR_QUIZ = 15


def generate_comprehension_quiz(text: str) -> List[ComprehensionQuestion]:
    """Невеликий тест на розуміння прочитаного (1-3 питання, кількість обирає
    AI сам залежно від того, скільки матеріалу дає уривок) для короткого
    тексту (~80-200 слів, читання на швидкість). На відміну від
    generate_questions() в services/test_generator.py — той розбиває ВЕЛИКИЙ
    текст книги на N частин і просить по одному питанню на частину; тут же
    весь (короткий) уривок іде в один виклик моделі одразу."""
    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_QUIZ:
        logger.info(f"Text too short for a comprehension quiz ({word_count} words), skipping")
        return []

    prompt = get_comprehension_quiz_prompt(text)

    try:
        raw_questions = get_json_from_response(generate_answer(prompt))
    except Exception as e:
        logger.error(f"Failed to generate comprehension quiz: {e}")
        return []

    if not isinstance(raw_questions, list):
        logger.error("Comprehension quiz response is not a JSON array")
        return []

    questions: List[ComprehensionQuestion] = []
    for raw_question in raw_questions[:3]:
        try:
            questions.append(ComprehensionQuestion(**raw_question))
        except ValidationError as e:
            logger.warning(f"Skipping malformed comprehension quiz question: {e}")

    return questions
