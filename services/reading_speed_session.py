import json
import logging
import wave
from pathlib import Path

from minio.error import S3Error
from vosk import KaldiRecognizer

from core.config import settings
from models.reading_speed import ReadingSpeedMetrics, ReferenceText, WordEvent, WordStatus
from services import substitution_check
from services.alignment import ReadingAlignmentTracker
from services.minio_client import minio_client
from services.vosk_service import SAMPLE_RATE, UNKNOWN_TOKEN, get_model
from utils.reference_text import build_grammar_phrase

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2  # PCM16
# Префікс об'єктів у тому самому books-бакеті, що й файли книг (Java,
# MinioService) — щоб не заводити окремий бакет лише заради аудіо сесій.
AUDIO_MINIO_PREFIX = "reading-audio"


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
        self._audio_object_key: str | None = None

        grammar = json.dumps([build_grammar_phrase(reference), UNKNOWN_TOKEN], ensure_ascii=False)
        self._recognizer = KaldiRecognizer(get_model(), SAMPLE_RATE, grammar)
        self._recognizer.SetWords(True)
        self._latest_partial_word_count = 0

        # Другий розпізнавач без граматики для гібридної перевірки замін —
        # у власному потоці, щоб не затримувати підсвітку слів.
        self._open_vosk = (
            substitution_check.OpenVoskWorker(get_model()) if substitution_check.is_available() else None
        )

    def process_audio_chunk(self, chunk: bytes) -> list[WordEvent]:
        self._audio_chunks.append(chunk)
        self._total_audio_bytes += len(chunk)
        if self._open_vosk is not None:
            self._open_vosk.feed(chunk)
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

    def attempted_reference_words(self) -> list[str]:
        """Тільки ті слова еталону, до яких дитина реально дочитала (без
        "хвоста" NOT_REACHED) — саме цей, коротший текст треба передавати в
        шар перевірки наголосу (forced alignment), інакше транскрипт (усі
        300-340 слів уривка) не відповідає тому, що фізично є в ~60-секундному
        записі, і MFA падає з NoAlignmentsError. Валідно лише після finalize()."""
        n = self._tracker.attempted_word_count()
        return [w.word for w in self.reference.words[:n]]

    def attempted_text(self) -> str:
        """Той самий прочитаний відрізок, що й attempted_reference_words(), але
        з оригінальним регістром і пунктуацією (display, не нормалізоване
        word) — саме такий текст варто давати AI для генерації тесту на
        розуміння прочитаного (services/comprehension_quiz.py). Валідно лише
        після finalize()."""
        n = self._tracker.attempted_word_count()
        return " ".join(w.display for w in self.reference.words[:n])

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
        self._upload_audio_to_minio()
        return events, metrics, self._tracker.all_events()

    def run_substitution_check(self) -> tuple[list[WordEvent], ReadingSpeedMetrics, list[WordEvent]]:
        """Гібридна перевірка замін після finalize(): слова, які граматика
        зарахувала як правильні, але обидва незалежні розпізнавачі без
        граматики (Vosk і Moonshine) почули інакше, перепозначаються як
        помилка або пропуск. Повертає (змінені події, оновлені метрики,
        повний пословний результат). Блокуючий виклик (Moonshine ~0,05 від
        тривалості запису) — запускати через asyncio.to_thread."""
        duration = self._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        if self._open_vosk is None:
            return [], self._tracker.compute_metrics(duration), self._tracker.all_events()

        open_words = self._open_vosk.finish()
        moonshine_words = substitution_check.transcribe_moonshine(b"".join(self._audio_chunks))
        attempted = self.attempted_reference_words()
        confirmed = substitution_check.confirmed_errors(attempted, open_words, moonshine_words)

        events = []
        for index, status in sorted(confirmed.items()):
            if self._tracker.status_of(index) == WordStatus.CORRECT:
                events.append(self._tracker.override_status(index, status))
        logger.info(
            f"Session {self.session_id}: substitution check changed {len(events)} word(s) "
            f"(both open recognizers disagree on {len(confirmed)})"
        )
        return events, self._tracker.compute_metrics(duration), self._tracker.all_events()

    def close(self) -> None:
        """Зупинити фоновий розпізнавач, якщо сесію перервано до finalize()."""
        if self._open_vosk is not None:
            self._open_vosk.stop()

    def audio_object_key(self) -> str | None:
        """MinIO object key аудіо сесії (для запису в attempt.audioFileName,
        щоб історія могла його відтворити) — None, якщо завантаження в MinIO
        не вдалося (не критично: локальний .wav все одно є, шар перевірки
        наголосу від нього не залежить, історія просто не покаже плеєр для
        цього attempt)."""
        return self._audio_object_key

    def _save_audio(self) -> None:
        path = self.audio_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(b"".join(self._audio_chunks))

        logger.info(f"Збережено аудіо сесії {self.session_id}: {path}")

    def _upload_audio_to_minio(self) -> None:
        """Копія аудіо в MinIO (той самий books-бакет, що й у Java) — потрібна
        для відтворення в історії читання. Локальний файл (audio_path())
        лишається на диску незалежно від успіху цього кроку — саме він
        використовується шаром перевірки наголосу (services/stress/
        pipeline.py), тому падіння завантаження в MinIO не повинно ламати
        решту сесії, лише логуємо і не заповнюємо audio_object_key."""
        object_key = f"{AUDIO_MINIO_PREFIX}/{self.session_id}.wav"
        try:
            minio_client.fput_object(
                bucket_name=settings.minio_bucket,
                object_name=object_key,
                file_path=str(self.audio_path()),
                content_type="audio/wav",
            )
            self._audio_object_key = object_key
        except S3Error as e:
            logger.error(f"Не вдалося завантажити аудіо сесії {self.session_id} в MinIO: {e}")
