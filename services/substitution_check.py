"""
Гібридна перевірка замін слів поверх граматики Vosk.

Проблема (master/recordings_v2, розділ 2.1 диплома): граматика з повного тексту
дає WER 1,8 % і ловить усі пропуски слів, але підтягує іншу форму слова до слова
еталону ("теплий" замість "теплу", "оцінили" замість "оцінило") — знайдено лише
2 з 16 таких замін у дорослих і 0 з 5 у дитини.

Рішення (той самий принцип, що в recordings_v2/score_v2.py, "Гібрид"): заміну
зараховуємо, якщо ОБИДВА незалежні розпізнавачі без граматики почули не те
слово — Vosk без граматики (працює паралельно під час читання, у власному
потоці, не затримуючи підсвітку) і Moonshine Base UK (запускається один раз
після читання на записі сесії). Розпізнавачі помиляються в різних місцях, тож
їхній збіг відсікає більшість хибних спрацювань: на записах recordings_v2
знайдено 95 % помилок читання проти 47 % лише з граматикою, 3,6 хибних на 100 слів.

Чому Moonshine Base, а не точніший Whisper large-v3-turbo (1,9 хибних на 100 слів):
на CPU сервера turbo працює ~30 с після 40-секундного читання (RTF 0,75, faster-whisper
int8) проти ~5 с у Moonshine Base (RTF 0,13); частка виявлених помилок однакова.
Модель задається MOONSHINE_MODEL_ID; на сервері з GPU доцільно перейти на turbo.

Сполучники "й"/"і" вважаються одним словом (чергування за милозвучністю).
"""

import json
import logging
import queue
import threading

import numpy as np
from vosk import KaldiRecognizer

from core.config import settings
from models.reading_speed import WordStatus
from services.alignment import _levenshtein_align
from services.vosk_service import SAMPLE_RATE
from utils.reference_text import normalize_text

logger = logging.getLogger(__name__)

# Сполучники, які в мовленні вільно чергуються і не є помилкою читання.
EQUIVALENT_WORDS = {"й": "і"}
# Whisper/Moonshine-подібні моделі погано працюють з довгими фрагментами
# (feature extractor Whisper узагалі обрізає вхід до 30 с), тому запис
# ріжеться по паузах на шматки не довші за це значення.
MAX_SEGMENT_SECONDS = 25.0

_moonshine = None  # (processor, model)
_lock = threading.Lock()


def canonical(words: list[str]) -> list[str]:
    return [EQUIVALENT_WORDS.get(w, w) for w in words]


# ------------------------------------------------------------ Moonshine
def load_model() -> None:
    """Завантажує Moonshine при старті застосунку. Відсутність torch/transformers
    або помилка завантаження не валить застосунок — гібрид просто вимикається."""
    global _moonshine
    if not settings.hybrid_check_enabled or _moonshine is not None:
        return
    try:
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        torch.set_num_threads(settings.moonshine_torch_threads)
        logger.info(f"Loading Moonshine model {settings.moonshine_model_id} (substitution check)")
        processor = AutoProcessor.from_pretrained(settings.moonshine_model_id)
        model = MoonshineForConditionalGeneration.from_pretrained(settings.moonshine_model_id).eval()
        _moonshine = (processor, model)
        logger.info("Moonshine model loaded")
    except Exception as e:  # ImportError, мережа, відсутня модель у кеші
        logger.warning(f"Substitution check disabled — Moonshine not available: {e}")


def is_available() -> bool:
    return settings.hybrid_check_enabled and _moonshine is not None


def _split_on_pauses(audio: np.ndarray, max_len: float = MAX_SEGMENT_SECONDS) -> list[tuple[int, int]]:
    """Межі шматків запису (у семплах), розрізаних у найдовшій паузі в другій
    половині кожного вікна max_len секунд."""
    hop = int(0.02 * SAMPLE_RATE)
    if len(audio) <= max_len * SAMPLE_RATE:
        return [(0, len(audio))]
    rms = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2) + 1e-12) for i in range(0, len(audio), hop)])
    quiet = rms < np.percentile(rms, 20) * 1.5
    segments, start = [], 0
    while (len(audio) - start) / SAMPLE_RATE > max_len:
        lo, hi = start + int(0.5 * max_len * SAMPLE_RATE), start + int(max_len * SAMPLE_RATE)
        best, best_len, run = None, 0, 0
        for f in range(lo // hop, min(hi // hop, len(quiet))):
            run = run + 1 if quiet[f] else 0
            if run >= best_len and run > 0:
                best_len, best = run, f
        cut = (best - best_len // 2) * hop if best is not None else hi
        segments.append((start, cut))
        start = cut
    segments.append((start, len(audio)))
    return segments


def transcribe_moonshine(pcm16: bytes) -> list[str]:
    import torch

    processor, model = _moonshine
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    words: list[str] = []
    for a, b in _split_on_pauses(audio):
        segment = audio[a:b]
        inputs = processor(segment, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        max_length = max(int(len(segment) * 8 / SAMPLE_RATE), 50)
        with _lock, torch.no_grad():
            ids = model.generate(**inputs, max_length=max_length)
        words += normalize_text(processor.decode(ids[0], skip_special_tokens=True)).split()
    return words


# ------------------------------------------------------------ Vosk без граматики
class OpenVoskWorker:
    """Другий розпізнавач Vosk (без граматики) на тій самій моделі, що працює в
    окремому потоці: чанки лише кладуться в чергу, тож обробка WS-повідомлень і
    підсвітка слів не чекають на нього."""

    _STOP = object()

    def __init__(self, model):
        self._recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        self._queue: queue.Queue = queue.Queue()
        self._parts: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        self._queue.put(chunk)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                break
            if self._recognizer.AcceptWaveform(item):
                self._parts.append(json.loads(self._recognizer.Result()).get("text", ""))
        self._parts.append(json.loads(self._recognizer.FinalResult()).get("text", ""))

    def finish(self, timeout: float = 60.0) -> list[str]:
        """Дочекатися обробки всіх чанків і повернути розпізнані слова."""
        self._queue.put(self._STOP)
        self._thread.join(timeout)
        text = " ".join(p for p in self._parts if p)
        return [w for w in normalize_text(text).split() if w not in ("unk",)]

    def stop(self) -> None:
        self._queue.put(self._STOP)


# ------------------------------------------------------------ правило гібриду
def _flagged(reference: list[str], hypothesis: list[str]) -> dict[int, WordStatus]:
    """Слова еталону, які за цим розпізнавачем прочитано з помилкою або пропущено."""
    flags: dict[int, WordStatus] = {}
    for op, ref_pos, _ in _levenshtein_align(canonical(reference), canonical(hypothesis)):
        if op == "sub":
            flags[ref_pos] = WordStatus.ERROR
        elif op == "del":
            flags[ref_pos] = WordStatus.SKIPPED
    return flags


def confirmed_errors(reference: list[str], open_vosk: list[str], moonshine: list[str]) -> dict[int, WordStatus]:
    """Індекси слів еталону, помилку в яких незалежно почули обидва розпізнавачі.
    Статус SKIPPED – лише якщо обидва вважають слово пропущеним, інакше ERROR."""
    a, b = _flagged(reference, open_vosk), _flagged(reference, moonshine)
    result = {}
    for i in a.keys() & b.keys():
        both_skipped = a[i] == WordStatus.SKIPPED and b[i] == WordStatus.SKIPPED
        result[i] = WordStatus.SKIPPED if both_skipped else WordStatus.ERROR
    return result
