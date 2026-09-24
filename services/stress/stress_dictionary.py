"""
Довідник правильного (еталонного) наголосу для слів української мови.

Перенесено як є з дипломних експериментів (`master/stress_dictionary.py`,
див. docs/reading-speed-implementation-overview.md та
STRESS_WORKLOG.md/STRESS_DETECTION.md у master/) — розділ 7
STRESS_WORKLOG.md прямо перелічує цей файл як один із трьох обов'язкових
модулів для перенесення в застосунок без змін.

Використовує пакет ukrainian-word-stress у легкому dictionary-only режимі
(без Stanza/PyTorch — лише словник у marisa-trie, кілька мегабайт, працює
повністю офлайн):

    pip install --no-deps ukrainian-word-stress
    pip install marisa-trie

Слово-омограф (за́мок / замо́к) без контексту однозначно не розв'язати,
тому для таких слів OnAmbiguity.All повертає ОБИДВА варіанти наголосу —
відповідь моделі вважається правильною, якщо вона співпала хоча б з
одним із них (це стосується й слів із варіативним наголосом на кшталт
по́милка/поми́лка — обидва варіанти нормативні).
"""

from ukrainian_word_stress import Disambiguation, OnAmbiguity, Stressifier, StressSymbol

_COMBINING_ACUTE = "́"
VOWELS = set("аеєиіїоуюя")

_stressify = Stressifier(
    disambiguation=Disambiguation.Dictionary,
    stress_symbol=StressSymbol.CombiningAcuteAccent,
    on_ambiguity=OnAmbiguity.All,
)


def count_vowels(word: str) -> int:
    return sum(1 for ch in word.lower() if ch in VOWELS)


def reference_stress_indices(word: str) -> set[int] | None:
    """
    Повертає множину допустимих номерів наголошеного складу (1-based,
    рахуючи голосні зліва направо). None — слова немає у словнику
    (не вдалося визначити еталонний наголос, слово пропускається
    з перевірки).
    """
    word = word.lower()
    n_vowels = count_vowels(word)
    if n_vowels == 0:
        return None
    if n_vowels == 1:
        return {1}

    marked = _stressify(word)
    chars = list(marked)
    indices = set()
    vowel_no = 0
    for i, ch in enumerate(chars):
        if ch in VOWELS:
            vowel_no += 1
            if i + 1 < len(chars) and chars[i + 1] == _COMBINING_ACUTE:
                indices.add(vowel_no)
    return indices or None
