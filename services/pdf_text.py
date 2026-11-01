"""
Витяг «чистого» тексту твору з PDF, який завантажив вчитель.

PyPDF2.extract_text() віддає все, що є на сторінці: колонтитули, номери
сторінок, виноски, номери виносок, приклеєні до слів («каве1»), підписи до
ілюстрацій, завдання й теорію з підручника. Усе це потрапляло і в еталонний
текст перевірки читання (учень цього не читає — пропуски й помилки, хибна
оцінка), і в текст для генерації тестів.

Тут текст береться з геометрії й шрифтів (PyMuPDF, rawdict), без прив'язки
до конкретної книги:

  * основний кегль книги — той, яким набрано найбільше символів; лишаємо
    тільки рядки цього кегля (виноски, підписи, завдання зазвичай дрібніші,
    заголовки й колонтитули — більші);
  * символи, значно дрібніші за основний кегль, — верхні індекси (номери
    виносок) — викидаємо посимвольно;
  * текст виносок і підписів окремо не шукаємо: вони набрані дрібніше й
    відсікаються за кеглем (спроба вгадувати виноску за цифрою на початку
    блоку викидала справжні абзаци — «строях⁴. Оце він…»);
  * рядки біля верхнього/нижнього краю сторінки, що повторюються на багатьох
    сторінках (без урахування цифр), або складаються лише з номерів, —
    колонтитули;
  * «пробіли» нульової ширини посеред слова («не бом» = «небом»);
  * наголошені голосні, які підручники набирають латинськими літерами
    (à, è, ³ …) або з комбінованим наголосом, — замінюємо на звичайні;
  * перенос слова через дефіс на межі рядка склеюємо;
  * абзаци — за абзацним відступом або збільшеним інтервалом.

Результат — список сторінок; абзаци всередині сторінки розділені "\n".
Абзац, що переходить на наступну сторінку, лишається розірваним на межі
сторінок — для речень і тестів це не заважає.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

# Рядок вважаємо основним текстом, якщо його кегль у межах ±7% від основного.
BODY_SIZE_TOLERANCE = 0.07
# Символ дрібніший за 80% основного кегля — верхній індекс / виноска.
SUPERSCRIPT_RATIO = 0.8
# Смуги біля верхнього та нижнього краю сторінки, де шукаємо колонтитули.
MARGIN_BAND = 0.1
# Скільки сторінок беремо для статистики (кегль, колонтитули), крім запитаних.
STATS_SAMPLE_PAGES = 12
# Рядок біля краю, що повторюється хоча б на стількох сторінках вибірки, — колонтитул.
REPEATED_LINE_MIN_PAGES = 3

# Наголошені голосні, набрані латинськими/службовими літерами шрифтів підручників.
_STRESSED_VOWELS = {
    "à": "а", "á": "а", "è": "и", "é": "е", "å": "е", "ì": "і", "í": "і", "³": "і",
    "ò": "о", "ó": "о", "ù": "у", "ú": "у", "ý": "у", "ÿ": "я", "þ": "ю", "º": "є", "¿": "ї",
}
_CYRILLIC = re.compile(r"[а-яіїєґ’'А-ЯІЇЄҐ]")
_COMBINING_ACUTE = "́"
_ONLY_NUMBERS_RE = re.compile(r"^[\d\s\-–—.,|]*$")


@dataclass
class _Line:
    text: str
    size: float          # домінантний кегль рядка
    x0: float
    y0: float
    y1: float
    block_id: int


def extract_pages(path: str, start_idx: int, end_idx: int) -> list[str]:
    """Чистий текст сторінок [start_idx, end_idx) (індекси з нуля)."""
    with pymupdf.open(path) as doc:
        end_idx = min(end_idx, len(doc))
        if start_idx >= end_idx:
            return []
        wanted = range(start_idx, end_idx)
        stats_pages = sorted(set(wanted) | set(_sample_pages(len(doc))))
        lines_by_page = {i: _page_lines(doc[i]) for i in stats_pages}
        heights = {i: doc[i].rect.height for i in stats_pages}

    body_size = _body_size(lines_by_page.values())
    if body_size is None:
        return ["" for _ in wanted]
    repeated = _repeated_margin_lines(lines_by_page, heights)

    pages = []
    for i in wanted:
        kept = [
            line for line in lines_by_page[i]
            if _is_body(line, body_size)
            and not _is_margin_noise(line, heights[i], repeated)
        ]
        pages.append(_to_paragraphs(kept, body_size))
    return pages


def extract_text(path: str, start_idx: int, end_idx: int) -> str:
    return "\n".join(page for page in extract_pages(path, start_idx, end_idx) if page)


def _sample_pages(total: int) -> list[int]:
    if total <= STATS_SAMPLE_PAGES:
        return list(range(total))
    step = total / STATS_SAMPLE_PAGES
    return [int(k * step) for k in range(STATS_SAMPLE_PAGES)]


def _page_lines(page) -> list[_Line]:
    raw = page.get_text("rawdict", sort=True)
    # Спершу збираємо всі символи, щоб знати кегль сторінки для верхніх індексів.
    sizes = Counter()
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                sizes[round(span["size"], 1)] += len(span["chars"])
    if not sizes:
        return []
    page_body = sizes.most_common(1)[0][0]
    small = page_body * SUPERSCRIPT_RATIO

    lines = []
    for block_id, block in enumerate(raw["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = []  # (символ, x0)
            weight = Counter()
            for span in line["spans"]:
                if span["size"] < small:
                    continue
                for c in span["chars"]:
                    chars.append((c["c"], c["bbox"][0]))
                    if not c["c"].isspace():
                        weight[round(span["size"], 1)] += 1
            if not weight:
                continue
            text = _join_chars(chars)
            if not text.strip():
                continue
            lines.append(_Line(
                text=text,
                size=weight.most_common(1)[0][0],
                x0=line["bbox"][0],
                y0=line["bbox"][1],
                y1=line["bbox"][3],
                block_id=block_id,
            ))
    return lines


def _join_chars(chars: list[tuple[str, float]]) -> str:
    out = []
    for i, (ch, x0) in enumerate(chars):
        if ch == " ":
            # Пробіл нульової ширини: наступна літера стоїть там само — це не пробіл.
            nxt = next((x for c, x in chars[i + 1:] if c != " "), None)
            if nxt is not None and nxt - x0 < 1.0:
                continue
        out.append(ch)
    return "".join(out)


def _body_size(pages) -> float | None:
    sizes = Counter()
    for lines in pages:
        for line in lines:
            sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _is_body(line: _Line, body_size: float) -> bool:
    return abs(line.size - body_size) <= body_size * BODY_SIZE_TOLERANCE


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip().lower()


def _in_margin(line: _Line, height: float) -> bool:
    return line.y1 < height * MARGIN_BAND or line.y0 > height * (1 - MARGIN_BAND)


def _repeated_margin_lines(lines_by_page: dict[int, list[_Line]], heights: dict[int, float]) -> set[str]:
    seen = Counter()
    for i, lines in lines_by_page.items():
        keys = {_margin_key(l.text) for l in lines if _in_margin(l, heights[i])}
        seen.update(keys)
    threshold = max(REPEATED_LINE_MIN_PAGES, len(lines_by_page) // 3)
    return {key for key, n in seen.items() if n >= threshold}


def _is_margin_noise(line: _Line, height: float, repeated: set[str]) -> bool:
    if _ONLY_NUMBERS_RE.match(line.text):
        return True
    return _in_margin(line, height) and _margin_key(line.text) in repeated


def _normalize_chars(text: str) -> str:
    text = text.replace(_COMBINING_ACUTE, "")
    # Гліф наголошеної голосної інколи відділено пробілом: «Кайм àк», «образ è».
    # Латинська à/è тощо не буває окремим українським словом, тож пробіл зайвий.
    glyphs = "".join(_STRESSED_VOWELS)
    text = re.sub(rf"(?<=[а-яіїєґ’]) ([{glyphs}])", r"\1", text)
    out = list(text)
    for i, ch in enumerate(out):
        if ch in _STRESSED_VOWELS:
            prev = out[i - 1] if i > 0 else ""
            nxt = out[i + 1] if i + 1 < len(out) else ""
            if _CYRILLIC.match(prev) or _CYRILLIC.match(nxt):
                out[i] = _STRESSED_VOWELS[ch]
    text = "".join(out)
    # Дефіс, закодований як «`» (Ні`ні, по`дитячому).
    return re.sub(r"(?<=\w)`(?=\w)", "-", text)


def _to_paragraphs(lines: list[_Line], body_size: float) -> str:
    if not lines:
        return ""
    # Лівий край основного тексту сторінки — найменший x0, на якому починається
    # хоча б кілька рядків (обтікання картинки дає окрему «колонку»).
    starts = Counter(round(l.x0) for l in lines)
    frequent = [x for x, n in starts.items() if n >= 2] or list(starts)
    left = min(frequent)

    paragraphs: list[list[str]] = []
    prev: _Line | None = None
    for line in lines:
        indent = line.x0 - left
        new_paragraph = (
            prev is None
            or body_size * 0.8 < indent < body_size * 4
            or line.y0 - prev.y1 > body_size * 0.9
        )
        if new_paragraph:
            paragraphs.append([line.text])
        else:
            paragraphs[-1].append(line.text)
        prev = line

    return "\n".join(_join_lines(p) for p in paragraphs)


def _join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    text = _normalize_chars(text)
    text = re.sub(r"\s+([,.!?;:…»])", r"\1", text)
    return re.sub(r"«\s+", "«", text)
