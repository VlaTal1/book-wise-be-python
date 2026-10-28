import json
import logging
import os
import time
from typing import Dict, List, Any, Union

from PyPDF2 import PdfReader
from minio import Minio
from minio.error import S3Error

from core.config import settings
from services.llm_service import ContentBlockedError, generate_answer
from services.rabbitmq_producer import RabbitMQProducer
from utils import split_text_into_parts, get_json_from_response
from utils.prompt_utils import get_prompt

logger = logging.getLogger(__name__)

minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure
)

rabbitmq_producer = RabbitMQProducer()


async def process_test_generation_request(request: Union[str, Dict[str, Any]]) -> None:
    try:
        if isinstance(request, str):
            try:
                request_data = json.loads(request)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load JSON: {e}")
                return
        else:
            request_data = request

        file_name = request_data.get('fileName')
        test_id = request_data.get('testId')
        start_page = request_data.get('startPage')
        end_page = request_data.get('endPage')
        question_count = request_data.get('questionCount')

        if not file_name or not test_id or not start_page or not end_page or not question_count:
            logger.error("Missing required parameters in request")
            return

        logger.info(f"Processing test generation request for book: {file_name}")

        book_text = await get_book_text(file_name, start_page, end_page)

        if not book_text:
            logger.error(f"Failed to get book's text: {file_name}")
            await send_error_response(file_name, test_id, f"Failed to get book's text: {file_name}")
            return

        questions = generate_questions(book_text, question_count)

        response = {
            "fileName": file_name,
            "testId": test_id,
            "questions": questions,
        }

        await send_response(response)

    except Exception as e:
        logger.error(f"Failed to process test generation request: {e}")
        if 'file_name' in locals() and file_name:
            # testId потрібен Java, щоб позначити тести як FAILED, а не лишити їх у стані генерації.
            await send_error_response(file_name, locals().get('test_id'), f"Failed to process test generation request: {str(e)}")


async def get_book_text(file_name: str, start_page: int, end_page: int) -> str:
    try:
        os.makedirs(settings.temp_dir, exist_ok=True)

        temp_file_path = os.path.join(settings.temp_dir, f"{file_name}.pdf")

        if not minio_client.bucket_exists(settings.minio_bucket):
            # Помилку з testId надішле викликач, коли отримає порожній текст.
            logger.warning(f"MinIO bucket {settings.minio_bucket} does not exists")
            return ""

        logger.info(f"Loading book {file_name} from MinIO")
        minio_client.fget_object(
            bucket_name=settings.minio_bucket,
            object_name=file_name,
            file_path=temp_file_path
        )

        logger.info(f"Retrieving text from pages {start_page} - {end_page}")
        reader = PdfReader(temp_file_path)

        if start_page < 1:
            start_page = 1
        if end_page > len(reader.pages):
            end_page = len(reader.pages)

        text = ""
        for page_num in range(start_page - 1, end_page):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"

        os.remove(temp_file_path)

        return text
    except S3Error as e:
        logger.error(f"MinIO error while retrieving book {file_name}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Failed to process PDF {file_name}: {e}")
        return ""


QUESTION_ATTEMPTS = 3


# Одна невдала відповідь моделі (порожня, без JSON, мережа) не повинна валити
# весь тест — пробуємо ще раз. Блокування за вмістом не повторюємо: на той
# самий текст Gemini відповість так само.
def generate_question(prompt: str, index: int) -> Dict[str, Any]:
    for attempt in range(1, QUESTION_ATTEMPTS + 1):
        try:
            return get_json_from_response(generate_answer(prompt))
        except ContentBlockedError:
            raise
        except Exception as e:
            logger.warning(f"Question {index}: attempt {attempt}/{QUESTION_ATTEMPTS} failed: {e}")
            if attempt == QUESTION_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


# Якщо фрагмент заблоковано (PROHIBITED_CONTENT тощо — це не вимикається
# налаштуваннями безпеки), пробуємо кожну половину окремо: проблемне місце
# зазвичай в одному абзаці. Не вийшло — пропускаємо це питання.
def generate_question_for_part(part: str, index: int) -> Union[Dict[str, Any], None]:
    try:
        return generate_question(get_prompt(part), index)
    except ContentBlockedError as e:
        logger.warning(f"Question {index}: part blocked ({e}), trying its halves")

    paragraphs = part.split("\n")
    if len(paragraphs) < 2:
        return None
    middle = len(paragraphs) // 2
    for half in ("\n".join(paragraphs[:middle]), "\n".join(paragraphs[middle:])):
        try:
            return generate_question(get_prompt(half), index)
        except ContentBlockedError as e:
            logger.warning(f"Question {index}: half blocked ({e})")
    return None


def generate_questions(text: str, count: int) -> List[Dict[str, Any]]:
    logger.info(f"Starting test generation")
    questions = []

    parts = split_text_into_parts(text, count)
    for index, part in enumerate(parts, start=1):
        logger.info(f"Generating question {index}/{count}")
        question = generate_question_for_part(part, index)
        if question is None:
            logger.warning(f"Question {index}: skipped, text blocked by Gemini")
            continue
        questions.append(question)

    if not questions:
        raise RuntimeError("All parts of the text were blocked by Gemini")

    logger.info(f"Test generation completed")

    return questions


async def send_response(response: Dict[str, Any]) -> None:
    try:
        logger.info(f"Sending results for book {response['fileName']}")

        await rabbitmq_producer.connect()

        await rabbitmq_producer.send_message(
            exchange_name='',
            routing_key=settings.rabbitmq_response_queue,
            message=json.dumps(response)
        )

        logger.info(f"Results for book {response['fileName']} sent successfully")

    except Exception as e:
        logger.error(f"Failed to send results: {e}")


async def send_error_response(file_name: str, test_id: Any, error_message: str) -> None:
    response = {
        "fileName": file_name,
        "testId": test_id,
        "error": error_message,
        "questions": []
    }
    await send_response(response)
