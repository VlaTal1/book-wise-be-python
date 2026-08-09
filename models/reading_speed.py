from enum import Enum

from pydantic import BaseModel


class WordStatus(str, Enum):
    CORRECT = "correct"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_IN_VOCABULARY = "not_in_vocabulary"


class ReferenceWord(BaseModel):
    index: int
    word: str
    in_vocabulary: bool


class ReferenceText(BaseModel):
    id: str
    title: str
    words: list[ReferenceWord]


class WordEvent(BaseModel):
    type: str = "word_event"
    index: int
    status: WordStatus
    # Чорнове попереднє припущення (ще не "дозріле" через TAIL_HOLDBACK) —
    # клієнт показує його як менш насичену/тимчасову підсвітку і замінює
    # справжнім (tentative=False) word_event, коли він прийде для того ж
    # index. Дає миттєвий візуальний відгук без очікування підтвердження.
    tentative: bool = False


class ReadingSpeedMetrics(BaseModel):
    total_words: int
    correct_count: int
    error_count: int
    skipped_count: int
    not_in_vocabulary_count: int
    duration_seconds: float
    wpm: float
    accuracy: float


class ReadingSpeedResult(BaseModel):
    type: str = "result"
    text_id: str
    metrics: ReadingSpeedMetrics
