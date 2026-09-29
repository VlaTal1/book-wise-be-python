"""
Випадковий уривок (~300-340 слів) з книги, призначеної дитині (Access у
Java, отримуємо через java_client.get_participant_books), для сесії читання
замість фіксованого resources/reference_texts/text_1.json.

Пайплайн: список призначених книг (Java, internal-ендпоінт) -> фільтр на
PDF (EPUB поки не парситься ніде в проекті — немає готового екстрактора
тексту) -> випадкова книга -> завантаження з MinIO -> PyPDF2, випадкова
сторінка -> очищення артефактів PDF (перенос слів через дефіс на межі рядка,
номери сторінок) -> розбиття на речення -> випадкове вікно речень такого
розміру, щоб не розрізати речення посередині.

Якщо книга не підійшла (замало тексту, скановані сторінки без шару тексту
тощо) — пробуємо наступну книгу зі списку, а не одразу здаємось.
"""

import logging
import os
import random
import re
import uuid

from PyPDF2 import PdfReader
from minio import Minio
from minio.error import S3Error

from core.config import settings
from models.reading_speed import ReferenceText
from services import java_client
from utils.reference_text import build_reference_text

logger = logging.getLogger(__name__)

minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)

# "так на 300, може чуть більше" (постановка задачі) — беремо вікно речень,
# що покриває [MIN, HARD_CAP] слів, віддаючи перевагу ближчому до MIN, але
# не розрізаючи останнє речення посередині (звідси запас до HARD_CAP).
EXCERPT_TARGET_MIN_WORDS = 300
EXCERPT_TARGET_MAX_WORDS = 340
EXCERPT_HARD_CAP_WORDS = 400
# Скільки сторінок від випадкової стартової читаємо за одну спробу — з
# запасом над тим, скільки типово треба тексту прози, щоб набрати 300+ слів
# (при цьому не читаємо книгу цілком, це занадто повільно для WS-хендшейку).
EXCERPT_MAX_PAGES_SCAN = 20

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\d{1,4}$")


class NoExcerptAvailableError(Exception):
    pass


def select_random_excerpt(participant_id: int) -> ReferenceText:
    books = java_client.get_participant_books(participant_id)
    pdf_books = [b for b in books if b.get("fileType") == "application/pdf"]
    random.shuffle(pdf_books)

    for book in pdf_books:
        try:
            excerpt = _extract_excerpt_from_book(book)
        except Exception:
            logger.exception(f"Failed to extract excerpt from book {book.get('id')} ({book.get('fileName')})")
            continue
        if excerpt is not None:
            return excerpt
        logger.warning(f"Book {book.get('id')} ({book.get('fileName')}) did not yield a usable excerpt, trying next")

    raise NoExcerptAvailableError(
        f"No suitable book excerpt for participant {participant_id}: "
        f"{len(books)} books assigned, {len(pdf_books)} of them PDF"
    )


def _extract_excerpt_from_book(book: dict) -> ReferenceText | None:
    file_name = book.get("fileName")
    book_id = book.get("id")
    if not file_name or book_id is None:
        return None
    title = book.get("title") or "Книга"

    local_path = _download_book(file_name)
    try:
        reader = PdfReader(local_path)
        num_pages = len(reader.pages)
        if num_pages == 0:
            return None

        start_page_idx = random.randrange(num_pages)
        raw_text = _extract_pages_text(reader, start_page_idx, EXCERPT_MAX_PAGES_SCAN)
        # Випадкова сторінка могла впасти близько до кінця книги — тоді
        # решти тексту може не вистачити на цільову кількість слів. Одна
        # додаткова спроба з початку книги дешевша й простіша, ніж шукати
        # "останню придатну" стартову сторінку заздалегідь.
        if len(raw_text.split()) < EXCERPT_TARGET_MIN_WORDS and start_page_idx > 0:
            start_page_idx = 0
            raw_text = _extract_pages_text(reader, 0, EXCERPT_MAX_PAGES_SCAN)

        cleaned = _clean_pdf_text(raw_text)
        sentences = _split_sentences(cleaned)
        if not sentences:
            return None

        window_text = _pick_excerpt_window(sentences)
        if window_text is None:
            return None

        excerpt_id = f"book:{book_id}:{start_page_idx}"
        excerpt_title = f"{title} (уривок)"
        return build_reference_text(excerpt_id, excerpt_title, window_text)
    except S3Error as e:
        logger.error(f"MinIO error while downloading book {file_name}: {e}")
        return None
    finally:
        _cleanup_temp_file(local_path)


def _download_book(file_name: str) -> str:
    os.makedirs(settings.temp_dir, exist_ok=True)
    local_path = os.path.join(settings.temp_dir, f"excerpt_{uuid.uuid4().hex}_{file_name}")
    minio_client.fget_object(
        bucket_name=settings.minio_bucket,
        object_name=file_name,
        file_path=local_path,
    )
    return local_path


def _cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _extract_pages_text(reader: PdfReader, start_page_idx: int, max_pages: int) -> str:
    end_idx = min(len(reader.pages), start_page_idx + max_pages)
    parts = [reader.pages[i].extract_text() or "" for i in range(start_page_idx, end_idx)]
    return "\n".join(parts)


def _clean_pdf_text(raw: str) -> str:
    """PyPDF2.extract_text() зберігає перенос слова через дефіс на межі
    рядка ("склад-\\nка") та часто виносить номер сторінки окремим рядком —
    жодне з цього не чистить existing normalize_text() (розрахований на вже
    чисту прозу), тому обробляємо тут, до розбиття на речення."""
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", raw)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or _PAGE_NUMBER_LINE_RE.fullmatch(stripped):
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    # Відкидаємо занадто короткі "речення" (1-2 слова) — типово це залишки
    # PDF-артефактів (розірвані заголовки, колонтитули), не повноцінна проза.
    return [p for p in parts if len(p.split()) >= 3]


def _pick_excerpt_window(sentences: list[str]) -> str | None:
    word_counts = [len(s.split()) for s in sentences]
    n = len(sentences)
    starts = list(range(n))
    random.shuffle(starts)

    for start in starts:
        total = 0
        end = start
        while end < n and total < EXCERPT_TARGET_MIN_WORDS:
            total += word_counts[end]
            end += 1
        if EXCERPT_TARGET_MIN_WORDS <= total <= EXCERPT_HARD_CAP_WORDS:
            return " ".join(sentences[start:end])
    return None
