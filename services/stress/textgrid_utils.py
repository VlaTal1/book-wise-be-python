"""
Мінімальний парсер Praat TextGrid (long format), у якому Montreal Forced
Aligner (MFA) зберігає результати forced alignment. Без зовнішніх
залежностей — формат простий і стабільний
(`mfa align --output_format long_textgrid`, дефолт).

Перенесено як є з `master/textgrid_utils.py` (обов'язковий модуль,
STRESS_WORKLOG.md, розділ 7).
"""

import re

_INTERVAL_RE = re.compile(
    r'intervals \[\d+\]:\s*'
    r'xmin = ([\d.]+)\s*'
    r'xmax = ([\d.]+)\s*'
    r'text = "(.*?)"',
    re.DOTALL,
)


def parse_textgrid(path: str) -> dict[str, list[tuple[str, float, float]]]:
    """Повертає {ім'я_тіру: [(текст, xmin, xmax), ...]} для кожного IntervalTier."""
    content = open(path, encoding="utf-8").read()

    tier_starts = [(m.start(), m.group(1)) for m in re.finditer(r'name = "(\w+)"', content)]
    tier_starts.append((len(content), None))

    tiers = {}
    for (start, name), (end, _) in zip(tier_starts, tier_starts[1:]):
        block = content[start:end]
        intervals = [
            (text, float(xmin), float(xmax))
            for xmin, xmax, text in _INTERVAL_RE.findall(block)
        ]
        tiers[name] = intervals
    return tiers
