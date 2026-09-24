from pydantic import BaseModel


class StressWordVerdict(BaseModel):
    index: int
    word: str
    # False, якщо слово не могло бути діагностоване (один склад, немає у
    # словнику наголосів, або MFA не знайшов очікувану к-сть голосних фонем
    # — реальна редукція/OOV у швидкій мові, ~15-22% слів за оцінкою з
    # STRESS_WORKLOG.md, п.6 розділу 5). Не помилка коду.
    checked: bool
    correct: bool | None = None
    predicted_syllable: int | None = None
    reference_syllables: list[int] | None = None


class StressCheckResult(BaseModel):
    # "done" — є хоч одне перевірене слово; "not_applicable" — жодного
    # діагностичного слова в тексті (замалий/непридатний уривок).
    status: str
    accuracy: float | None
    checked_words: int
    correct_words: int
    words: list[StressWordVerdict]
