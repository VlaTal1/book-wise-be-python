import logging

from vosk import Model, SetLogLevel

from core.config import settings

logger = logging.getLogger(__name__)

SetLogLevel(-1)

SAMPLE_RATE = 16000
UNKNOWN_TOKEN = "<UNK>"

_model: Model | None = None


def load_model() -> Model:
    global _model
    if _model is None:
        logger.info(f"Loading Vosk model from {settings.vosk_model_path}")
        _model = Model(settings.vosk_model_path)
        logger.info("Vosk model loaded")
    return _model


def get_model() -> Model:
    if _model is None:
        raise RuntimeError("Vosk model not loaded yet — call load_model() at startup")
    return _model
