import json
import re
from functools import lru_cache
from pathlib import Path

from core.config import settings
from models.reading_speed import ReferenceText, ReferenceWord
from services.vocabulary import is_in_vocabulary


# Слово — пробіг \w, всередині якого може бути апостроф: «п’ять», «б’ється»
# — одне слово, а не «п» + «ять». Дефіс і решта пунктуації слова розділяють
# ('далеко-широко' -> 'далеко широко', а не неіснуюче 'далекошироко').
# Апостроф лише між літерами, тож лапки-одинарні 'так' його не зачіпають.
_WORD_RE = re.compile(r"\w+(?:['’ʼ]\w+)*")
# У словниках Vosk (graph/words.txt) і MFA апостроф — ASCII ', а в текстах
# книг і в розпізнаванні Moonshine трапляються ’ та ʼ.
_APOSTROPHE_RE = re.compile(r"[’ʼ]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_word(word: str) -> str:
    return _APOSTROPHE_RE.sub("'", word.lower())


def normalize_text(text: str) -> str:
    """Нижній регістр, апостроф -> ', пунктуація (в т.ч. дефіс) -> пробіл."""
    return " ".join(normalize_word(w) for w in _WORD_RE.findall(text))


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
        # Перенос рядка між словом і тире репліки («дітвори.\n— А») інакше
        # потрапляє всередину плашки слова, і тире малюється під ним.
        display = _WHITESPACE_RE.sub(" ", raw_text[m.start():end].rstrip())
        result.append((normalize_word(m.group()), display))
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
