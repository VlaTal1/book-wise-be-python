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


# \w+ дає ті самі межі слів, що й normalize_text() (пунктуацію/дефіс/апостроф
# замінено на пробіл, отже слова — це максимальні пробіги \w) — але без
# попереднього lower()/заміни пунктуації, тому можна одночасно з тим самим
# словом дістати "як воно виглядає в оригіналі" для показу на екрані.
_WORD_RE = re.compile(r"\w+")


def _tokenize_with_display(raw_text: str) -> list[tuple[str, str]]:
    """Повертає [(нормалізоване_слово, текст_для_показу), ...] в тому самому
    порядку й з тими самими межами слів, що дав би normalize_text(raw_text)
    .split(" ") — це важливо: нормалізовані слова йдуть у граматику Vosk і
    словник наголосів, змінювати їхні межі тут не можна.

    текст_для_показу — оригінальний фрагмент (регістр + пунктуація, що йде
    ОДРАЗУ ЗА словом: кома/крапка/тире) — лише для читабельного відображення
    на екрані читання, не бере участі в розпізнаванні. Пунктуація ПЕРЕД
    першим словом усього тексту (напр. відкриваюча лапка) губиться — це
    прийнятний, рідкісний косметичний компроміс."""
    matches = list(_WORD_RE.finditer(raw_text))
    result = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        display = raw_text[m.start():end].rstrip()
        result.append((m.group().lower(), display))
    return result


def build_reference_text(text_id: str, title: str, raw_text: str) -> ReferenceText:
    """Спільна логіка перетворення сирого тексту (з фіксованого JSON АБО з
    динамічно обраного уривка книги, див. services/book_excerpt.py) на
    ReferenceText з перевіркою покриття словником Vosk по кожному слову."""
    words = []
    for raw_word, display in _tokenize_with_display(raw_text):
        in_vocab = is_in_vocabulary(raw_word, settings.vosk_model_path)
        words.append(ReferenceWord(index=len(words), word=raw_word, in_vocabulary=in_vocab, display=display))

    return ReferenceText(id=text_id, title=title, words=words)


@lru_cache(maxsize=8)
def load_reference_text(text_id: str) -> ReferenceText:
    path = Path(settings.reference_texts_dir) / f"{text_id}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return build_reference_text(data["id"], data.get("title", data["id"]), data["text"])


def build_grammar_phrase(reference: ReferenceText) -> str:
    """Граматика Vosk як ЦІЛА фраза (не список слів) — саме цей формат дав
    1.8% WER у бенчмарку, на порядок кращий за 'мішок слів' (розділ 3.1 дока)."""
    return " ".join(w.word for w in reference.words)
