import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "Test Generation Service"

    # RabbitMQ settings
    rabbitmq_host: str = Field(default=os.getenv("RABBITMQ_HOST", "localhost"))
    rabbitmq_port: int = Field(default=int(os.getenv("RABBITMQ_PORT", 5672)))
    rabbitmq_user: str = Field(default=os.getenv("RABBITMQ_USER", "guest"))
    rabbitmq_password: str = Field(default=os.getenv("RABBITMQ_PASSWORD", "guest"))
    rabbitmq_request_queue: str = Field(default=os.getenv("RABBITMQ_REQUEST_QUEUE", "test_generation_request"))
    rabbitmq_exchange: str = Field(default=os.getenv("RABBITMQ_EXCHANGE", "test_generation_exchange"))
    rabbitmq_response_queue: str = Field(default=os.getenv("RABBITMQ_RESPONSE_QUEUE", "test_generation_response"))
    rabbitmq_routing_key: str = Field(default=os.getenv("RABBITMQ_ROUTING_KEY", "generate_test"))

    # MinIO settings
    minio_endpoint: str = Field(default=os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    minio_access_key: str = Field(default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = Field(default=os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = Field(default=os.getenv("MINIO_BUCKET", "books"))
    minio_secure: bool = Field(default=bool(int(os.getenv("MINIO_SECURE", "0"))))

    # LLM для генерування тестів (services/llm_service.py). Модель обрано в
    # master/quiz_exp: найкраще дотримується правил запиту щодо кількості запитань.
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "gemini-3.8-flash"))
    llm_timeout_seconds: int = Field(default=int(os.getenv("LLM_TIMEOUT_SECONDS", 90)))

    # Temporary file storage
    temp_dir: str = Field(default=os.getenv("TEMP_DIR", "/tmp"))

    # Supabase (JWT auth for the reading-speed WebSocket)
    supabase_url: str = Field(default=os.getenv("SUPABASE_URL", ""))

    # Reading-speed feature (Vosk guided decoding)
    vosk_model_path: str = Field(default=os.getenv("VOSK_MODEL_PATH", "vosk_models/vosk-model-uk-v3-lgraph"))
    reference_texts_dir: str = Field(default=os.getenv("REFERENCE_TEXTS_DIR", "resources/reference_texts"))
    reading_sessions_audio_dir: str = Field(default=os.getenv("READING_SESSIONS_AUDIO_DIR", "reading_sessions"))

    # Java backend (service-to-service call to save reading-speed results)
    java_base_url: str = Field(default=os.getenv("JAVA_BASE_URL", "http://localhost:8080"))
    java_internal_api_key: str = Field(default=os.getenv("JAVA_INTERNAL_API_KEY", ""))

    # Stress-check layer (окремий асинхронний шар поверх reading-speed,
    # запускається ПІСЛЯ фіналізації WS-сесії — див. services/
    # stress_background.py, docs/reading-speed-implementation-overview.md).
    # Модель, натренована в master/train_stress_model_final.py на
    # MFA-ознаках (energy/duration/pitch/phone-identity), серіалізована
    # через joblib — рантайм лише завантажує й застосовує її.
    stress_model_path: str = Field(default=os.getenv("STRESS_MODEL_PATH", "resources/models/stress_model.joblib"))
    # Montreal Forced Aligner — важка runtime-залежність (conda-оточення з
    # Kaldi/OpenFST, не pip-пакет). mfa_bin_dir додається на початок PATH
    # перед викликом (за зразком master/STRESS_WORKLOG.md, розділ 3, п.9);
    # порожній рядок означає "mfa вже в системному PATH".
    mfa_bin_dir: str = Field(default=os.getenv("MFA_BIN_DIR", ""))
    mfa_acoustic_model: str = Field(default=os.getenv("MFA_ACOUSTIC_MODEL", "ukrainian_mfa"))
    mfa_dictionary: str = Field(default=os.getenv("MFA_DICTIONARY", "ukrainian_mfa"))
    # Kaldi-дефолти (beam=10, retry_beam=40) розраховані на короткі утterance
    # (секунди), а не на ~60-секундний одноблочний запис сесії читання —
    # на реальних мобільних записах (тихіше/шумніше за дипломні контрольні
    # записи) це стабільно валилось з NoAlignmentsError. Значення нижче —
    # саме ті, що рекомендує власне повідомлення про цю помилку MFA
    # ("mfa align ... --beam 100 --retry_beam 400"), підтверджено вручну.
    mfa_beam: int = Field(default=int(os.getenv("MFA_BEAM", 100)))
    mfa_retry_beam: int = Field(default=int(os.getenv("MFA_RETRY_BEAM", 400)))
    mfa_align_timeout_seconds: int = Field(default=int(os.getenv("MFA_ALIGN_TIMEOUT_SECONDS", 120)))
    # Робочі директорії для forced alignment (окремий піддиректорій на
    # сесію, не видаляється автоматично — корисно для діагностики; можна
    # прибирати періодичним cron/скриптом при потребі).
    stress_mfa_corpus_dir: str = Field(default=os.getenv("STRESS_MFA_CORPUS_DIR", "stress_sessions/corpus"))
    stress_mfa_output_dir: str = Field(default=os.getenv("STRESS_MFA_OUTPUT_DIR", "stress_sessions/output"))

    # Гібридна перевірка замін слів (services/substitution_check.py).
    # Граматика Vosk підтягує іншу форму слова ("теплий" замість "теплу") до
    # слова еталону й не бачить таких замін (master/recordings_v2: 2 з 16 у
    # дорослих, 0 з 5 у дитини). Тому заміну додатково підтверджують два
    # незалежні розпізнавачі без граматики: Vosk (паралельно під час читання)
    # і Moonshine (після читання). Якщо torch/transformers не встановлені або
    # модель не завантажилась — перевірка вимикається, працює лише граматика.
    hybrid_check_enabled: bool = Field(default=os.getenv("HYBRID_CHECK_ENABLED", "true").lower() == "true")
    moonshine_model_id: str = Field(default=os.getenv("MOONSHINE_MODEL_ID", "UsefulSensors/moonshine-base-uk"))
    moonshine_torch_threads: int = Field(default=int(os.getenv("MOONSHINE_TORCH_THREADS", 4)))

    @property
    def rabbitmq_url(self) -> str:
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"


settings = Settings()
