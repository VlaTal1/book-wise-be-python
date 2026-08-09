import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_vocabulary(words_txt_path: str) -> frozenset[str]:
    words = set()
    with open(words_txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                words.add(parts[0])
    logger.info(f"Loaded {len(words)} words from Vosk vocabulary: {words_txt_path}")
    return frozenset(words)


def is_in_vocabulary(word: str, model_dir: str) -> bool:
    """Перевіряє, чи слово покрите словником моделі Vosk (graph/words.txt).
    Слова поза словником неможливо розпізнати навіть з грамматикою — див.
    docs/reading-speed-feature-design.md, розділ 3.1."""
    words_txt = str(Path(model_dir) / "graph" / "words.txt")
    return word in _load_vocabulary(words_txt)
