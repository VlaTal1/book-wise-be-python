"""
Спільна логіка видобування ознак наголосу по точних (MFA) межах голосних
фонем. Перенесено з `master/stress_features.py` (обов'язковий модуль,
STRESS_WORKLOG.md розділ 7), з двома відмінностями від дипломної версії:

- Константи F0 (раніше імпортувались з `stress_acoustic.py`, застарілої
  Vosk-евристики, яку свідомо НЕ переносимо в застосунок — розділ 7
  STRESS_WORKLOG.md) інлайновані тут напряму.
- Додано `phone_onehot()` — фіча ідентичності фонеми, яка в дипломі
  тестувалась лише ad hoc у чаті (не влита в жоден .py-файл, TODO п.2
  розділу 5 STRESS_WORKLOG.md) і дала найкращий задокументований
  результат (74.0%/80.4%/86.1% на трьох незалежних тестах). PHONE_LIST
  МУСИТЬ лишатись у тому самому порядку, в якому навчалась модель
  (див. train_stress_model_final.py у master/, там же збережено
  stress_model.joblib) — порядок визначає порядок стовпців one-hot.
"""

import numpy as np

VOWEL_PHONES = {"a", "ɑ", "ɐ", "e", "ɛ", "i", "iː", "ɪ", "o", "ɔ", "u", "ʊ"}

PHONE_LIST = ["a", "ɑ", "ɐ", "e", "ɛ", "i", "iː", "ɪ", "o", "ɔ", "u", "ʊ"]

F0_MIN_HZ = 75.0
F0_MAX_HZ = 400.0
F0_VOICING_THRESHOLD = 0.3


def vowel_phones_in_span(
    phones: list[tuple[str, float, float]], wstart: float, wend: float
) -> list[tuple[str, float, float]]:
    return [(t, s, e) for t, s, e in phones if s >= wstart - 1e-6 and e <= wend + 1e-6 and t in VOWEL_PHONES]


def frame_f0_scalar(audio: np.ndarray, sr: int) -> float:
    """Середній F0 (Гц) по озвучених кадрах короткого сегмента; 0, якщо жодного озвученого."""
    frame_len = min(len(audio), max(1, int(sr * 0.025)))
    hop_len = max(1, int(sr * 0.010))
    lag_min = max(1, int(sr / F0_MAX_HZ))
    lag_max = int(sr / F0_MIN_HZ)
    window = np.hanning(frame_len) if frame_len > 1 else np.ones(1)

    voiced_f0 = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop_len):
        seg = audio[start:start + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        seg = seg * window
        ac = np.correlate(seg, seg, mode="full")[frame_len - 1:]
        ac0 = ac[0] + 1e-12
        hi = min(lag_max, len(ac) - 1)
        if hi <= lag_min:
            continue
        peak_lag = lag_min + int(np.argmax(ac[lag_min:hi]))
        if ac[peak_lag] / ac0 > F0_VOICING_THRESHOLD:
            voiced_f0.append(sr / peak_lag)
    return float(np.mean(voiced_f0)) if voiced_f0 else 0.0


def extract_syllable_features(
    vowel_phones: list[tuple[str, float, float]], audio: np.ndarray, sr: int
) -> dict:
    energy, duration, pitch, phone_id = [], [], [], []
    for t, s, e in vowel_phones:
        seg = audio[int(s * sr):int(e * sr)]
        energy.append(float(np.sqrt(np.mean(seg**2) + 1e-12)) if len(seg) else 0.0)
        duration.append(e - s)
        pitch.append(frame_f0_scalar(seg, sr))
        phone_id.append(t)
    return {"energy": energy, "duration": duration, "pitch": pitch, "phone_id": phone_id}


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else np.zeros_like(x)


def phone_onehot(phone_id: str) -> list[float]:
    return [1.0 if phone_id == p else 0.0 for p in PHONE_LIST]
