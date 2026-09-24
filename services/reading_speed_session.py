import json
import logging
import wave
from pathlib import Path

from vosk import KaldiRecognizer

from core.config import settings
from models.reading_speed import ReadingSpeedMetrics, ReferenceText, WordEvent
from services.alignment import ReadingAlignmentTracker
from services.vosk_service import SAMPLE_RATE, UNKNOWN_TOKEN, get_model
from utils.reference_text import build_grammar_phrase

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2  # PCM16


class ReadingSpeedSession:
    """Один сеанс читання: один WS-конект = одна грамматика (весь еталонний
    текст однією фразою, розділ 3.1 дока) + один трекер вирівнювання + буфер
    аудіо для збереження на диск (рішення по п.3 в docs/reading-speed-feature-
    design.md, розділ 8)."""

    def __init__(self, session_id: str, user_id: str, reference: ReferenceText):
        self.session_id = session_id
        self.user_id = user_id
        self.reference = reference
        self._tracker = ReadingAlignmentTracker(reference.words)
        self._audio_chunks: list[bytes] = []
        self._total_audio_bytes = 0

        grammar = json.dumps([build_grammar_phrase(reference), UNKNOWN_TOKEN], ensure_ascii=False)
        self._recognizer = KaldiRecognizer(get_model(), SAMPLE_RATE, grammar)
        self._recognizer.SetWords(True)
        self._latest_partial_word_count = 0

    def process_audio_chunk(self, chunk: bytes) -> list[WordEvent]:
        self._audio_chunks.append(chunk)
        self._total_audio_bytes += len(chunk)
        self._recognizer.AcceptWaveform(chunk)
        partial = json.loads(self._recognizer.PartialResult())
        words = partial.get("partial", "").split()
        self._latest_partial_word_count = len(words)
        committed_events = self._tracker.feed_partial(words)
        # Чорнові прогнози для ще не підтверджених слів — миттєвий відгук на
        # клієнті, поки TAIL_HOLDBACK тримає остаточне рішення (розділ 9, п.5
        # дока: "способи зменшити затримку" — варіант 2).
        tentative_events = self._tracker.preview_tentative()
        return committed_events + tentative_events

    def audio_path(self) -> Path:
        return Path(settings.reading_sessions_audio_dir) / f"{self.session_id}.wav"

    def is_reading_complete(self) -> bool:
        """Розпізнавання (навіть ще не підтверджене через TAIL_HOLDBACK) вже
        дійшло до кінця еталонного тексту — сигнал форсувати finalize(), не
        чекаючи client-side "stop".

        Vosk PartialResult() структурно НІКОЛИ не показує саме останнє
        розпізнане слово всього мовлення — воно з'являється лише в
        FinalResult(). Тому й сира довжина partial, і буфер трекера (який
        будується з partial) назавжди застрягають на -1 від реальної
        кількості слів, скільки аудіо не давай — це підтверджено емпірично
        на vals-record.wav (partial доходив рівно до 112 з 113 слів і більше
        не рухався). Тож допускаємо запас в 1 слово в обох перевірках."""
        margin = 1
        total = len(self.reference.words)
        return (
            self._tracker.has_recognized_through_end(margin=margin)
            or self._latest_partial_word_count >= total - margin
        )

    def finalize(self) -> tuple[list[WordEvent], ReadingSpeedMetrics, list[WordEvent]]:
        """Повертає (events, metrics, all_word_events): events — лише те, що
        змінилося на цьому останньому кроці (для дотрансляції клієнту),
        all_word_events — повний пословний результат усієї сесії (для
        збереження історії в Java)."""
        final = json.loads(self._recognizer.FinalResult())
        words = final.get("text", "").split()
        events = self._tracker.finalize(words)

        # Тривалість рахуємо за кількістю фактично надісланих аудіо-семплів,
        # а не за wall-clock часом обробки на сервері — інакше мережеві
        # затримки/джиттер спотворювали б WPM.
        duration = self._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        metrics = self._tracker.compute_metrics(duration)
        self._save_audio()
        return events, metrics, self._tracker.all_events()

    def _save_audio(self) -> None:
        path = self.audio_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(b"".join(self._audio_chunks))

        logger.info(f"Збережено аудіо сесії {self.session_id}: {path}")
