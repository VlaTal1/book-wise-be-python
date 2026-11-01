"""
Заздалегідь підготовлені уривки книг каталогу (resources/catalog_excerpts/
<catalogKey>.json, збирає books/tools/build_excerpts.py). Для таких книг
випадковий уривок для читання береться з цього списку — кожен уривок є
закінченою сценою, а не довільним вікном речень з PDF.
"""

import json
import logging
import random
from functools import lru_cache
from pathlib import Path

from models.reading_speed import ReferenceText
from utils.reference_text import build_reference_text

logger = logging.getLogger(__name__)

EXCERPTS_DIR = Path(__file__).resolve().parent.parent / "resources" / "catalog_excerpts"


@lru_cache(maxsize=16)
def _load(catalog_key: str) -> dict | None:
    path = EXCERPTS_DIR / f"{catalog_key}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if data.get("excerpts") else None


def has_excerpts(catalog_key: str | None) -> bool:
    return bool(catalog_key) and _load(catalog_key) is not None


def random_excerpt(catalog_key: str, title: str | None = None) -> ReferenceText | None:
    data = _load(catalog_key)
    if data is None:
        return None
    excerpt = random.choice(data["excerpts"])
    book_title = title or data.get("title") or "Книга"
    return build_reference_text(f"catalog:{excerpt['id']}", f"{book_title} (уривок)", excerpt["text"])
