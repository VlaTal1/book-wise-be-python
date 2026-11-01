"""
Абзаци чистого тексту сторінок книги — для попереднього перегляду, коли
вчитель призначає класу фрагмент для читання на швидкість (Java викликає
POST /internal/book-paragraphs, вчитель обирає перший і останній абзац).

Абзац, що переходить на наступну сторінку, extract_pages() віддає двома
шматками — тут їх склеюємо, щоб вчитель бачив і міг обрати цілий абзац.
"""

import logging
import os
import re
import uuid

from core.config import settings
from services.minio_client import minio_client
from services.pdf_text import extract_pages

logger = logging.getLogger(__name__)

# Захист від випадкового «1–500»: попередній перегляд — це кілька сторінок.
MAX_PREVIEW_PAGES = 15

_SENTENCE_END = (".", "!", "?", "…", "»", ":", ";", ")")


def get_paragraphs(file_name: str, start_page: int, end_page: int) -> list[str]:
    """Сторінки рахуються з 1, end_page включно (як у тестів класу)."""
    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")
    if end_page - start_page + 1 > MAX_PREVIEW_PAGES:
        raise ValueError(f"Page range is limited to {MAX_PREVIEW_PAGES} pages")

    os.makedirs(settings.temp_dir, exist_ok=True)
    local_path = os.path.join(settings.temp_dir, f"paragraphs_{uuid.uuid4().hex}_{os.path.basename(file_name)}")
    minio_client.fget_object(bucket_name=settings.minio_bucket, object_name=file_name, file_path=local_path)
    try:
        pages = extract_pages(local_path, start_page - 1, end_page)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
    return merge_page_paragraphs(pages)


def merge_page_paragraphs(pages: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for page in pages:
        page_paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
        if not page_paragraphs:
            continue
        first = page_paragraphs[0]
        # Абзац продовжується з попередньої сторінки: попередній не закінчився
        # реченням, а цей починається з малої літери.
        if paragraphs and not paragraphs[-1].endswith(_SENTENCE_END) and first[:1].islower():
            paragraphs[-1] = f"{paragraphs[-1]} {first}"
            page_paragraphs = page_paragraphs[1:]
        paragraphs.extend(page_paragraphs)
    return [re.sub(r"\s+", " ", p) for p in paragraphs]
