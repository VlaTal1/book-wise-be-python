import asyncio
import logging

from fastapi import APIRouter, Header, HTTPException
from minio.error import S3Error
from pydantic import BaseModel, Field

from core.config import settings
from services.book_paragraphs import get_paragraphs

logger = logging.getLogger(__name__)
router = APIRouter()


class BookParagraphsRequest(BaseModel):
    file_name: str = Field(alias="fileName")
    start_page: int = Field(alias="startPage")
    end_page: int = Field(alias="endPage")


class BookParagraphsResponse(BaseModel):
    paragraphs: list[str]


# Викликає лише Java (service-to-service) — той самий внутрішній API-ключ,
# що й у зворотному напрямку (Python -> Java /internal/**).
@router.post("/book-paragraphs", response_model=BookParagraphsResponse)
async def book_paragraphs(
    request: BookParagraphsRequest,
    x_internal_api_key: str = Header(...),
):
    if not settings.java_internal_api_key or x_internal_api_key != settings.java_internal_api_key:
        raise HTTPException(status_code=403, detail="Invalid internal API key")
    try:
        paragraphs = await asyncio.to_thread(get_paragraphs, request.file_name, request.start_page, request.end_page)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except S3Error as e:
        logger.error(f"MinIO error while reading {request.file_name}: {e}")
        raise HTTPException(status_code=404, detail="Book file not found")
    return BookParagraphsResponse(paragraphs=paragraphs)
