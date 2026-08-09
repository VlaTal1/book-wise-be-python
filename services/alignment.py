from models.reading_speed import ReadingSpeedMetrics, ReferenceWord, WordEvent, WordStatus

# Наскільки широке вікно еталону порівнюємо з накопиченим буфером розпізнаних
# слів (запас над довжиною буфера, щоб DP міг знайти реальний пропуск, а не
# лише збіг 1:1). Заміряно емпірично на vals-record.wav (розділ 9, п.5 дока):
# при ALIGN_SLACK >= 4 разом з TAIL_HOLDBACK=2 повертається той самий баг
# хибного перестрибування вперед, що й раніше — тримати <= 3.
ALIGN_SLACK = 3

# Скільки останніх розпізнаних слів у буфері НЕ фіксуємо остаточно — рішення
# по них може змінитися, коли надійде більше контексту. Без цього наївне
# "жадібне" вирівнювання по одному слову хибно перестрибувало вперед по
# еталону після одноразової помилкової підміни (типово — одразу після
# not_in_vocabulary слова) і застрягало там до кінця сесії.
#
# Заміряно емпірично (той самий запис, той самий важкий кейс з OOV-словом):
# TAIL_HOLDBACK=1 стабільно ламає точність (та ж помилка), TAIL_HOLDBACK=2 —
# безпечний мінімум, зберігає точність (98.2%) і дає ~22% менше затримки
# підсвітки слів порівняно з TAIL_HOLDBACK=4 (6.6 слів замість 8.5). Нижче
# опускатися ризиковано без глибшого перегляду алгоритму (наприклад,
# розбиття грамматики на речення замість однієї фрази — Vosk сам по собі
# теж вносить частину цієї затримки, не тільки цей буфер).
TAIL_HOLDBACK = 2

# Скільки слів еталону вперед від self._pointer показуємо як "в процесі"
# (tentative) — навмисно маленьке число, це лише візуальний натяк, що зараз
# читається десь тут, а не спроба вгадати результат заздалегідь.
PREVIEW_LOOKAHEAD = 2

_Op = tuple[str, int | None, int | None]


class ReadingAlignmentTracker:
    """Порівнює потік розпізнаних Vosk слів з еталонним текстом у міру
    надходження (без очікування кінця сесії). Буферизує нові розпізнані слова
    і вирівнює їх з еталоном через Levenshtein-DP по вікну, фіксуючи лише
    "дозрілу" частину результату (усе, крім останніх TAIL_HOLDBACK слів
    буфера) — так реальний пропуск/помилка не плутається з тимчасовою
    невизначеністю через брак контексту."""

    def __init__(self, reference_words: list[ReferenceWord]):
        self._reference_words = reference_words
        self._pointer = 0
        # Скільки слів з ПОВНОГО (постійно зростаючого) потоку Vosk
        # partial/final результатів ми вже забрали в буфер — монотонно
        # зростає, НЕ зменшується, коли слова з буфера комітяться і
        # видаляються (інакше наступний виклик знову захопить те саме слово
        # вдруге — саме такий дубль і був справжньою причиною першого зламу).
        self._total_words_seen = 0
        self._pending_recognized: list[str] = []
        self._statuses: dict[int, WordStatus] = {}

    def feed_partial(self, recognized_words: list[str]) -> list[WordEvent]:
        """Викликається після кожного шматка аудіо з поточною (ще не
        остаточною) гіпотезою Vosk. Останнє слово гіпотези вважається
        нестабільним і не додається до буфера — Vosk може ще переглянути
        його в наступному partial-результаті."""
        stable_count = max(len(recognized_words) - 1, 0)
        new_words = recognized_words[self._total_words_seen:stable_count]
        self._pending_recognized.extend(new_words)
        self._total_words_seen += len(new_words)
        return self._flush(final=False)

    def finalize(self, recognized_words: list[str]) -> list[WordEvent]:
        """Викликається один раз з остаточною гіпотезою Vosk (FinalResult) —
        фіксує все, що залишилось у буфері, і позначає залишок еталону як
        пропущений/не розпізнаний."""
        new_words = recognized_words[self._total_words_seen:]
        self._pending_recognized.extend(new_words)
        self._total_words_seen += len(new_words)
        events = self._flush(final=True)

        events.extend(self._emit_not_in_vocabulary_run())
        while self._pointer < len(self._reference_words):
            events.append(self._set_status(self._pointer, WordStatus.SKIPPED))
            self._pointer += 1
        return events

    def has_recognized_through_end(self, margin: int = 0) -> bool:
        """Чи розпізнаний потік (включно з ще не підтвердженим буфером, який
        застряг через TAIL_HOLDBACK) уже покриває весь еталонний текст (з
        запасом `margin` слів — Vosk PartialResult() структурно ніколи не
        показує саме останнє розпізнане слово всього мовлення, тож без
        запасу перевірка недосяжна, поки не прийде FinalResult()).

        Без цієї перевірки сесія ніколи не завершується сама наприкінці
        тексту: останнім TAIL_HOLDBACK словам ніколи не набереться наступного
        контексту для коміту, бо після кінця тексту просто нема що ще
        розпізнавати — читачу довелося б повторювати останні слова, щоб
        сесія хоч колись завершилась. Тому кінець визначаємо не за
        підтвердженим станом (self._pointer), а за тим, що вже РОЗПІЗНАНО
        (pointer + розмір буфера), і форсуємо finalize() ззовні."""
        return self._pointer + len(self._pending_recognized) >= len(self._reference_words) - margin

    def preview_tentative(self) -> list[WordEvent]:
        """Показує наступні кілька слів еталону від поточної підтвердженої
        позиції як "в процесі" — навмисно НЕ через Levenshtein-вирівнювання по
        широкому вікну (як у _flush): перша версія рахувала прогноз тим самим
        широким вікном/DP, що й справжній коміт, і через це могла "забігти
        вперед" по тексту раніше, ніж читач реально дочитав до того слова —
        саме те, від чого захищає TAIL_HOLDBACK для справжніх комітів, але
        прогноз його не враховував. Ця версія прив'язана лише до self._pointer
        (який рухається лише при справжньому коміті) і показує щонайбільше
        PREVIEW_LOOKAHEAD слів вперед — фізично не може забігти далі."""
        if not self._pending_recognized:
            return []

        events = []
        idx = self._pointer
        shown = 0
        while idx < len(self._reference_words) and shown < PREVIEW_LOOKAHEAD:
            if self._reference_words[idx].in_vocabulary:
                events.append(WordEvent(index=idx, status=WordStatus.CORRECT, tentative=True))
                shown += 1
            idx += 1
        return events

    def all_events(self) -> list[WordEvent]:
        """Повний пословний результат (для збереження історії в Java) — має
        сенс викликати лише після finalize(), коли кожне слово еталону вже
        отримало якийсь статус."""
        return [
            WordEvent(index=i, status=self._statuses[i])
            for i in range(len(self._reference_words))
            if i in self._statuses
        ]

    def compute_metrics(self, duration_seconds: float) -> ReadingSpeedMetrics:
        correct = sum(1 for s in self._statuses.values() if s == WordStatus.CORRECT)
        error = sum(1 for s in self._statuses.values() if s == WordStatus.ERROR)
        skipped = sum(1 for s in self._statuses.values() if s == WordStatus.SKIPPED)
        not_in_vocab = sum(1 for s in self._statuses.values() if s == WordStatus.NOT_IN_VOCABULARY)
        total = len(self._reference_words)
        minutes = max(duration_seconds / 60, 1e-6)

        return ReadingSpeedMetrics(
            total_words=total,
            correct_count=correct,
            error_count=error,
            # "не розпізнано" зараховуємо як пропущене за замовчуванням (рішення
            # по п.4, docs/reading-speed-feature-design.md, розділ 8); окремо
            # зберігаємо not_in_vocabulary_count для майбутнього перерахунку.
            skipped_count=skipped + not_in_vocab,
            not_in_vocabulary_count=not_in_vocab,
            duration_seconds=round(duration_seconds, 2),
            wpm=round(correct / minutes, 1),
            accuracy=round(correct / total, 4) if total else 0.0,
        )

    def _build_window(self, pending_count: int) -> tuple[list[int], list[str]]:
        window_size = pending_count + ALIGN_SLACK
        ref_window_indices: list[int] = []
        idx = self._pointer
        while idx < len(self._reference_words) and len(ref_window_indices) < window_size:
            if self._reference_words[idx].in_vocabulary:
                ref_window_indices.append(idx)
            idx += 1
        ref_window_words = [self._reference_words[i].word for i in ref_window_indices]
        return ref_window_indices, ref_window_words

    def _flush(self, final: bool) -> list[WordEvent]:
        events = self._emit_not_in_vocabulary_run()
        if not self._pending_recognized:
            return events

        ref_window_indices, ref_window_words = self._build_window(len(self._pending_recognized))
        ops = _levenshtein_align(ref_window_words, self._pending_recognized)
        commit_upto = len(ops) if final else _find_commit_boundary(ops, TAIL_HOLDBACK)

        last_committed_window_pos = None
        committed_rec_count = 0
        for op, ref_pos, rec_pos in ops[:commit_upto]:
            if op == "match":
                events.append(self._set_status(ref_window_indices[ref_pos], WordStatus.CORRECT))
                last_committed_window_pos = ref_pos
            elif op == "sub":
                events.append(self._set_status(ref_window_indices[ref_pos], WordStatus.ERROR))
                last_committed_window_pos = ref_pos
            elif op == "del":
                events.append(self._set_status(ref_window_indices[ref_pos], WordStatus.SKIPPED))
                last_committed_window_pos = ref_pos
            if rec_pos is not None:
                committed_rec_count += 1

        if last_committed_window_pos is not None:
            new_pointer = ref_window_indices[last_committed_window_pos] + 1
            # Слова еталону поза словником, що потрапили в цей проміжок (їх
            # пропустили при побудові вікна вище, бо DP їх ніколи не бачить),
            # теж потрібно позначити — інакше вони лишаться без статусу.
            for skipped_idx in range(self._pointer, new_pointer):
                if skipped_idx not in self._statuses:
                    events.append(self._set_status(skipped_idx, WordStatus.NOT_IN_VOCABULARY))
            self._pointer = new_pointer

        self._pending_recognized = self._pending_recognized[committed_rec_count:]
        return events

    def _emit_not_in_vocabulary_run(self) -> list[WordEvent]:
        events = []
        while (
            self._pointer < len(self._reference_words)
            and not self._reference_words[self._pointer].in_vocabulary
        ):
            events.append(self._set_status(self._pointer, WordStatus.NOT_IN_VOCABULARY))
            self._pointer += 1
        return events

    def _set_status(self, index: int, status: WordStatus) -> WordEvent:
        self._statuses[index] = status
        return WordEvent(index=index, status=status)


def _levenshtein_align(ref_words: list[str], hyp_words: list[str]) -> list[_Op]:
    """Стандартне Levenshtein-вирівнювання зі зворотним проходом (той самий
    підхід, що й word_level_comparison у master/guided_benchmark.py, але як
    вікно потокового вирівнювання, а не одноразовий прохід по всьому тексту).
    Повертає список операцій (match/sub/del/ins) у прямому порядку."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    ops: list[_Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", None, j - 1))
            j -= 1
        else:
            ops.append(("del", i - 1, None))
            i -= 1
    ops.reverse()
    return ops


def _find_commit_boundary(ops: list[_Op], tail_holdback: int) -> int:
    """Скільки операцій з початку списку вважаємо остаточними — усе, крім
    тих, що торкаються останніх `tail_holdback` розпізнаних слів буфера."""
    rec_consuming_seen = 0
    for k in range(len(ops) - 1, -1, -1):
        if ops[k][2] is not None:
            rec_consuming_seen += 1
        if rec_consuming_seen > tail_holdback:
            return k + 1
    return 0
