import logging

import joblib

from core.config import settings

logger = logging.getLogger(__name__)

_bundle: dict | None = None


def load_model() -> None:
    """Завантажує serialized bundle {model, feature_columns, phone_list} з
    диску (див. master/train_stress_model_final.py). Викликається один раз
    при старті застосунку (як і Vosk-модель, services/vosk_service.py) —
    щоб перший запит на перевірку наголосу не чекав на завантаження моделі.

    На відміну від Vosk-моделі (без якої вся фіча читання не працює),
    відсутність файлу моделі тут НЕ повинна валити застосунок — шар
    перевірки наголосу є окремим, додатковим над основною фічею (розділ
    "Що робимо" у задачі), тож при відсутності моделі просто логуємо
    попередження й позначаємо шар недоступним (is_available() -> False,
    stress_background.py в цьому разі відрапортує FAILED і не зламає
    основний потік читання)."""
    global _bundle
    if _bundle is not None:
        return
    try:
        logger.info(f"Loading stress model from {settings.stress_model_path}")
        _bundle = joblib.load(settings.stress_model_path)
        logger.info("Stress model loaded")
    except (FileNotFoundError, OSError, EOFError) as e:
        logger.warning(f"Stress model not available ({settings.stress_model_path}): {e}. Stress-check layer disabled.")


def is_available() -> bool:
    return _bundle is not None


def get_bundle() -> dict:
    if _bundle is None:
        raise RuntimeError("Stress model not loaded — is_available() is False, see startup logs")
    return _bundle
