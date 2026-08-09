import json
import re
from functools import lru_cache
from pathlib import Path

from core.config import settings
from models.reading_speed import ReferenceText, ReferenceWord
from services.vocabulary import is_in_vocabulary


def normalize_text(text: str) -> str:
    """Той самий алгоритм нормалізації, що й у бенчмарках у master/ —
    пунктуацію (в т.ч. дефіс) заміняємо на пробіл, а не видаляємо, інакше
    'далеко-широко' склеюється в неіснуюче слово 'далекошироко'."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=8)
def load_reference_text(text_id: str) -> ReferenceText:
    path = Path(settings.reference_texts_dir) / f"{text_id}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    normalized = normalize_text(data["text"])
    words = []
    for raw_word in normalized.split(" "):
        if not raw_word:
            continue
        in_vocab = is_in_vocabulary(raw_word, settings.vosk_model_path)
        words.append(ReferenceWord(index=len(words), word=raw_word, in_vocabulary=in_vocab))

    return ReferenceText(id=data["id"], title=data.get("title", data["id"]), words=words)


def build_grammar_phrase(reference: ReferenceText) -> str:
    """Граматика Vosk як ЦІЛА фраза (не список слів) — саме цей формат дав
    1.8% WER у бенчмарку, на порядок кращий за 'мішок слів' (розділ 3.1 дока)."""
    return " ".join(w.word for w in reference.words)
