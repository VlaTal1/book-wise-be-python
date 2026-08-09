"""
Автономна перевірка пайплайну (Vosk streaming + guided grammar + alignment +
метрики), без WebSocket-транспорту і без мобілки — саме так, як задумано в
docs/reading-speed-feature-design.md, розділ 9, п.1.

Годує ReadingSpeedSession тим самим записом, який використовувався для
бенчмарків (master/records/vals-record.wav), шматками, що імітують потокове
надходження аудіо з мобілки, і друкує live-події підсвітки слів по мірі
надходження + фінальні метрики.

Запуск:
  venv/bin/python3 scripts/test_reading_speed_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.reading_speed_session import ReadingSpeedSession
from services.vosk_service import SAMPLE_RATE, load_model
from utils.reference_text import load_reference_text

AUDIO_PATH = Path(__file__).resolve().parent.parent.parent / "master" / "records" / "vals-record.wav"
CHUNK_MS = 200


def load_pcm16_16k_mono(path: Path) -> bytes:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        n_target = int(round(duration * SAMPLE_RATE))
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=n_target, endpoint=False)
        audio = np.interp(x_new, x_old, audio)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def main():
    if not AUDIO_PATH.exists():
        print(f"Аудіо не знайдено: {AUDIO_PATH}")
        return

    print("Завантаження моделі Vosk...")
    load_model()

    reference = load_reference_text("text_1")
    print(f"\nЕталонний текст '{reference.title}': {len(reference.words)} слів")
    not_in_vocab = [w.word for w in reference.words if not w.in_vocabulary]
    print(f"Слів поза словником моделі: {not_in_vocab}\n")

    pcm = load_pcm16_16k_mono(AUDIO_PATH)
    chunk_size = int(SAMPLE_RATE * 2 * CHUNK_MS / 1000)  # 2 байти на семпл (PCM16)

    session = ReadingSpeedSession(session_id="pipeline-test", user_id="test-user", reference=reference)

    t0 = time.perf_counter()
    for i in range(0, len(pcm), chunk_size):
        events = session.process_audio_chunk(pcm[i:i + chunk_size])
        for event in events:
            word = reference.words[event.index].word
            print(f"  [{event.status.value:>17}] #{event.index:>3} {word}")

    events, metrics, _all_word_events = session.finalize()
    for event in events:
        word = reference.words[event.index].word
        print(f"  [{event.status.value:>17}] #{event.index:>3} {word}")
    t1 = time.perf_counter()

    print("\n" + "=" * 60)
    print("ФІНАЛЬНІ МЕТРИКИ")
    print("=" * 60)
    print(metrics.model_dump_json(indent=2))
    print(f"\nЧас обробки (весь пайплайн): {t1 - t0:.2f}с")

    audio_path = Path("reading_sessions") / "pipeline-test.wav"
    print(f"Аудіо збережено: {audio_path.resolve()} (existss={audio_path.exists()})")


if __name__ == "__main__":
    main()
