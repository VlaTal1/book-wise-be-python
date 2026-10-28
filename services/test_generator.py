import json
import logging
import os
from typing import Dict, List, Any, Union

from PyPDF2 import PdfReader
from minio import Minio
from minio.error import S3Error

from core.config import settings
from services.llm_service import generate_answer
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
            await send_error_response(file_name, f"Failed to get book's text: {file_name}")
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
            await send_error_response(file_name, f"Failed to process test generation request: {str(e)}")


async def get_book_text(file_name: str, start_page: int, end_page: int) -> str:
    try:
        os.makedirs(settings.temp_dir, exist_ok=True)

        temp_file_path = os.path.join(settings.temp_dir, f"{file_name}.pdf")

        if not minio_client.bucket_exists(settings.minio_bucket):
            logger.warning(f"MinIO bucket {settings.minio_bucket} does not exists")
            await send_error_response(file_name, f"MinIO bucket {settings.minio_bucket} does not exists")

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


def generate_questions(text: str, count: int) -> List[Dict[str, Any]]:
    logger.info(f"Starting test generation")
    questions = []

    parts = split_text_into_parts(text, count)
    for index, part in enumerate(parts, start=1):
        logger.info(f"Generating question {index}/{count}")
        prompt = get_prompt(part)
        json_part = get_json_from_response(generate_answer(prompt))
        questions.append(json_part)

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


async def send_error_response(file_name: str, error_message: str) -> None:
    response = {
        "fileName": file_name,
        "error": error_message,
        "questions": []
    }
    await send_response(response)
