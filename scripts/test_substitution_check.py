"""
Перевірка гібридної перевірки замін на записах master/recordings_v2 (3 диктори,
20 записів) через справжню ReadingSpeedSession — так, як працює WS-сесія:
аудіо подається чанками по 64 мс, після finalize() запускається
run_substitution_check(). Порівнює знайдені помилки читання з ground truth
(сценарії + ручна розмітка + підтверджені відхилення) і міряє час.

Запуск: .venv/bin/python scripts/test_substitution_check.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.reading_speed import WordStatus  # noqa: E402
from services import substitution_check  # noqa: E402
from services.reading_speed_session import ReadingSpeedSession  # noqa: E402
from services.vosk_service import SAMPLE_RATE, load_model  # noqa: E402
from utils.reference_text import build_reference_text  # noqa: E402

DATA = ROOT.parent.parent.parent / "unik" / "diplom" / "diplom" / "master" / "recordings_v2"
CHUNK_BYTES = 2048  # ~64 мс PCM16, як у мобільному застосунку


def truth(name: str, n_words: int) -> set[int]:
    _, tid, mode = name.split("_")
    gt = json.load(open(DATA / "ground_truth.json", encoding="utf8"))[tid]
    errors: set[int] = set()
    if mode == "errors":
        sc = gt["scenario"]
        errors |= {e["index"] for e in sc["skip"]} | {e["index"] for e in sc["substitution"]}
    elif mode == "natural":
        for row in list(csv.reader(open(DATA / f"annotation_{tid}.csv", encoding="utf-8-sig"), delimiter=";"))[1:]:
            if row[2].strip().lower() in ("пропуск", "помилка"):
                errors.add(int(row[0]) - 1)
    dev = json.load(open(DATA / "deviations.json", encoding="utf8")).get(name, {})
    errors |= {d["index"] for kind in ("substitution", "skip") for d in dev.get(kind, [])}
    return errors


def main():
    load_model()
    substitution_check.load_model()
    assert substitution_check.is_available(), "Moonshine не завантажився"

    totals = {"grammar": [0, 0, 0], "hybrid": [0, 0, 0]}  # tp, fn, fp
    words_total, chunk_times, post_times, durations = 0, [], [], []
    for wav in sorted((DATA / "wav").glob("*.wav")):
        tid = wav.stem.split("_")[1]
        raw = (DATA / "texts" / f"{tid}.txt").read_text(encoding="utf8")
        reference = build_reference_text(tid, tid, raw)
        session = ReadingSpeedSession(session_id=f"test-{wav.stem}", user_id="test", reference=reference)
        session._upload_audio_to_minio = lambda: None  # без MinIO у тесті

        audio, sr = sf.read(str(wav), dtype="float32")
        assert sr == SAMPLE_RATE
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        durations.append(len(audio) / sr)
        for i in range(0, len(pcm), CHUNK_BYTES):
            t0 = time.perf_counter()
            session.process_audio_chunk(pcm[i:i + CHUNK_BYTES])
            chunk_times.append(time.perf_counter() - t0)

        _, _, grammar_events = session.finalize()
        grammar_err = {e.index for e in grammar_events if e.status in (WordStatus.ERROR, WordStatus.SKIPPED)}
        t0 = time.perf_counter()
        _, metrics, hybrid_events = session.run_substitution_check()
        post_times.append(time.perf_counter() - t0)
        hybrid_err = {e.index for e in hybrid_events if e.status in (WordStatus.ERROR, WordStatus.SKIPPED)}

        gt = truth(wav.stem, len(reference.words))
        words_total += len(reference.words)
        for key, det in (("grammar", grammar_err), ("hybrid", hybrid_err)):
            totals[key][0] += len(det & gt); totals[key][1] += len(gt - det); totals[key][2] += len(det - gt)
        print(f"{wav.stem:20s} помилок {len(gt):2d} | граматика {len(grammar_err & gt)}/{len(gt)} хибних {len(grammar_err - gt)} | "
              f"гібрид {len(hybrid_err & gt)}/{len(gt)} хибних {len(hybrid_err - gt)} | WPM {metrics.wpm} | після читання {post_times[-1]:.2f} с")

    print()
    for key, (tp, fn, fp) in totals.items():
        print(f"{key:8s} знайдено {tp}/{tp + fn} ({100 * tp / (tp + fn):.0f} %), хибних на 100 слів {100 * fp / words_total:.1f}")
    print(f"чанк (64 мс аудіо): середнє {1000 * np.mean(chunk_times):.1f} мс, 99-й перцентиль {1000 * np.percentile(chunk_times, 99):.1f} мс")
    print(f"очікування після читання: середнє {np.mean(post_times):.2f} с, максимум {max(post_times):.2f} с "
          f"(середня тривалість запису {np.mean(durations):.1f} с)")


if __name__ == "__main__":
    main()
