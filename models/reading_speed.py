from enum import Enum

from pydantic import BaseModel


class WordStatus(str, Enum):
    CORRECT = "correct"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_IN_VOCABULARY = "not_in_vocabulary"
    # Еталонний текст свідомо довший (300-340 слів), ніж встигає прочитати
    # дитина за виділену хвилину (запас на різний темп читання) — тому
    # "хвіст", до якого розпізнавання просто не дійшло до кінця сесії, не є
    # помилкою читання (на відміну від SKIPPED — реального пропуску слова
    # ВСЕРЕДИНІ прочитаного). NOT_REACHED виключається з підрахунку метрик
    # (ReadingAlignmentTracker.compute_metrics) — рахуємо точність лише по
    # тому, що дитина реально встигла прочитати.
    NOT_REACHED = "not_reached"


class ReferenceWord(BaseModel):
    index: int
    word: str
    in_vocabulary: bool
    # Оригінальний фрагмент тексту (регістр + пунктуація одразу за словом) —
    # лише для показу на екрані читання (utils/reference_text.py); word
    # (нормалізоване, без пунктуації) — те, що йде в граматику Vosk/словник
    # наголосів і бере участь у розпізнаванні/вирівнюванні.
    display: str


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


class ComprehensionAnswer(BaseModel):
    answer: str
    correct: bool


class ComprehensionQuestion(BaseModel):
    question: str
    quote: str
    answers: list[ComprehensionAnswer]


class ReadingSpeedResult(BaseModel):
    type: str = "result"
    text_id: str
    metrics: ReadingSpeedMetrics
    # Id збереженого ReadingSpeedAttempt у Java — мобілка використовує його,
    # щоб опитувати GET /api/reading-speed-attempts/{id} і показати
    # лоадер/прогрес-бар шару перевірки наголосу (рахується в фоні окремо,
    # див. services/stress_background.py). None, якщо збереження в Java не
    # вдалося — тоді шар наголосу теж не запускається.
    attempt_id: int | None = None
    # Невеликий тест на розуміння прочитаного (1-3 питання, кількість обирає
    # AI залежно від того, скільки матеріалу дав уривок) — генерується
    # синхронно, одразу тут, з тексту, який дитина РЕАЛЬНО встигла прочитати
    # (services/comprehension_quiz.py). None, якщо уривок закороткий для тесту
    # або генерація не вдалася — тоді мобілка просто не показує секцію тесту.
    quiz: list[ComprehensionQuestion] | None = None
