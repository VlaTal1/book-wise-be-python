"""
Єдина точка входу шару перевірки наголосу: "аудіо + еталонний текст ->
пословний вердикт по наголосу". Той самий контракт, що описаний як
орієнтир у STRESS_WORKLOG.md (розділ 7, п.3):

    check_stress_in_recording(audio_path, reference_text, model) -> list[dict]

Пайплайн (детальний опис — docs/reading-speed-implementation-overview.md,
розділ 4, і master/STRESS_WORKLOG.md, розділ 3):
  1. forced alignment (Montreal Forced Aligner, модель ukrainian_mfa) на
     повному аудіо сесії читання + еталонному тексті як транскрипті —
     точні межі кожної ГОЛОСНОЇ ФОНЕМИ (а не лише слова, як дає Vosk).
  2. для кожного "діагностичного" слова (2+ голосних, є у словнику
     наголосів, к-сть голосних фонем MFA == к-сть голосних букв —
     розбіжність означає реальну фонетичну редукцію, а не баг):
     видобуття ознак (energy/duration/pitch/phone identity) по кожній
     голосній.
  3. навчена модель (GradientBoostingClassifier, master/train_stress_model_
     final.py) -> ймовірність "цей склад наголошений" по кожному складу
     слова -> argmax.
  4. порівняння з нормативним наголосом (stress_dictionary).

Це навмисно НЕ real-time — forced alignment триває секунди на файл
(відкрите архітектурне питання, STRESS_WORKLOG.md розділ 7, вирішене на
користь окремого асинхронного шару обробки, а не вбудовування в WS-сесію
читання, див. services/stress_background.py).
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from core.config import settings
from models.stress import StressCheckResult, StressWordVerdict
from services.stress import model_store
from services.stress.stress_dictionary import count_vowels, reference_stress_indices
from services.stress.stress_features import extract_syllable_features, phone_onehot, vowel_phones_in_span, zscore
from services.stress.textgrid_utils import parse_textgrid

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], None]


@dataclass
class StressCheckError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _report(on_progress: ProgressCallback | None, stage: str, percent: int) -> None:
    if on_progress is not None:
        try:
            on_progress(stage, percent)
        except Exception:
            logger.exception(f"Progress callback failed at stage={stage}")


def _run_mfa_align(corpus_dir: Path, output_dir: Path) -> None:
    import os

    env = os.environ.copy()
    if settings.mfa_bin_dir:
        env["PATH"] = f"{settings.mfa_bin_dir}:{env.get('PATH', '')}"

    cmd = [
        "mfa", "align", str(corpus_dir),
        settings.mfa_acoustic_model, settings.mfa_dictionary, str(output_dir),
        "--clean", "-j", "1",
    ]
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=settings.mfa_align_timeout_seconds,
        )
    except FileNotFoundError as e:
        raise StressCheckError(
            "Montreal Forced Aligner (mfa) не знайдено — перевірте settings.mfa_bin_dir / PATH"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise StressCheckError(f"mfa align перевищив таймаут ({settings.mfa_align_timeout_seconds}с)") from e

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        raise StressCheckError(f"mfa align завершився з кодом {result.returncode}: {tail}")


def check_stress_in_recording(
    session_id: str,
    audio_path: Path,
    reference_words: list[str],
    on_progress: ProgressCallback | None = None,
) -> StressCheckResult:
    if not model_store.is_available():
        raise StressCheckError("Stress model not loaded — see startup logs (STRESS_MODEL_PATH)")

    corpus_dir = Path(settings.stress_mfa_corpus_dir) / session_id
    output_dir = Path(settings.stress_mfa_output_dir) / session_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_audio = corpus_dir / f"{session_id}.wav"
    if not corpus_audio.exists():
        corpus_audio.symlink_to(Path(audio_path).resolve())
    (corpus_dir / f"{session_id}.txt").write_text(" ".join(reference_words), encoding="utf-8")

    _report(on_progress, "aligning", 20)
    _run_mfa_align(corpus_dir, output_dir)

    textgrid_path = output_dir / f"{session_id}.TextGrid"
    if not textgrid_path.exists():
        raise StressCheckError("MFA не створив TextGrid — ймовірно, не зміг вирівняти аудіо з текстом")

    _report(on_progress, "extracting_features", 60)
    tiers = parse_textgrid(str(textgrid_path))
    words_tier = [(t, s, e) for t, s, e in tiers["words"] if t.strip()]
    phones_tier = tiers["phones"]

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    if len(words_tier) != len(reference_words):
        logger.warning(
            f"Session {session_id}: MFA words tier count ({len(words_tier)}) != "
            f"reference word count ({len(reference_words)}) — перевіряю по мінімуму"
        )

    bundle = model_store.get_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    _report(on_progress, "scoring", 80)

    verdicts: list[StressWordVerdict] = []
    n = min(len(words_tier), len(reference_words))
    for idx in range(n):
        word = reference_words[idx]
        _, wstart, wend = words_tier[idx]

        n_vowels = count_vowels(word)
        ref_indices = reference_stress_indices(word)
        if n_vowels <= 1 or ref_indices is None:
            verdicts.append(StressWordVerdict(index=idx, word=word, checked=False))
            continue

        vp = vowel_phones_in_span(phones_tier, wstart, wend)
        if len(vp) != n_vowels:
            verdicts.append(StressWordVerdict(index=idx, word=word, checked=False))
            continue

        feats = extract_syllable_features(vp, audio, sr)
        energy_z = zscore(np.array(feats["energy"]))
        duration_z = zscore(np.array(feats["duration"]))
        pitch_z = zscore(np.array(feats["pitch"]))
        syllable_count = len(feats["duration"])

        X = np.array([
            [energy_z[i], duration_z[i], pitch_z[i], i / max(1, syllable_count - 1), syllable_count]
            + phone_onehot(feats["phone_id"][i])
            for i in range(syllable_count)
        ])
        assert X.shape[1] == len(feature_columns)

        probs = model.predict_proba(X)[:, 1]
        predicted_syllable = int(np.argmax(probs)) + 1

        verdicts.append(StressWordVerdict(
            index=idx,
            word=word,
            checked=True,
            correct=predicted_syllable in ref_indices,
            predicted_syllable=predicted_syllable,
            reference_syllables=sorted(ref_indices),
        ))

    checked = [v for v in verdicts if v.checked]
    correct = [v for v in checked if v.correct]

    _report(on_progress, "finalizing", 95)

    return StressCheckResult(
        status="done" if checked else "not_applicable",
        accuracy=round(len(correct) / len(checked), 4) if checked else None,
        checked_words=len(checked),
        correct_words=len(correct),
        words=verdicts,
    )
