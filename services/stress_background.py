"""
Асинхронний запуск шару перевірки наголосу ПІСЛЯ того, як WS-сесія читання
вже завершена й результат показаний користувачу (потрібен повний
аудіофайл сесії — forced alignment не може працювати на потоці чанків,
див. docs/reading-speed-implementation-overview.md). Обробка триває у
фоні незалежно від того, чи користувач лишився на екрані результатів,
чи вийшов з екрана/застосунку (мобілка лише ОПИТУЄ Java за станом,
не тримає з'єднання) — саме тому прогрес доповідається в Java
(services/java_client.py), а не через WS/live push.

`asyncio.create_task()` без збереження референсу на задачу ризикує
збиранням сміття до завершення — тому тримаємо активні задачі в
module-level set (`_running_tasks`), яка звільняється в done-callback.
"""

import asyncio
import contextlib
import logging
from pathlib import Path

from services import java_client
from services.stress.pipeline import StressCheckError, check_stress_in_recording

logger = logging.getLogger(__name__)

_running_tasks: set = set()

# "aligning" (20%) — це фактично весь час forced alignment (subprocess
# `mfa align`, секунди-десятки секунд на файл), а сам MFA не дає жодного
# проміжного прогресу. Без цього тікера мобілка (опитує раз в
# STRESS_POLL_INTERVAL_MS=3с, mobile/readingSpeed/index.tsx) бачить прогрес-
# бар "застряглим" на 20% майже весь час обробки, а потім 60/80/95/100%
# проскакують за долі секунди між двома опитуваннями. Тікер лише імітує
# плавне заповнення — реальний прогрес (extracting_features/scoring/
# finalizing) все одно приходить з check_stress_in_recording і має пріоритет.
_PROGRESS_TICK_SECONDS = 2
_PROGRESS_TICK_STEP = 5
_PROGRESS_TICK_CAP = 55  # лишає запас під реальні 60/80/95, щоб не було стрибка назад


def schedule_stress_check(attempt_id: int, session_id: str, audio_path: Path, reference_words: list[str]) -> None:
    task = asyncio.create_task(_process(attempt_id, session_id, audio_path, reference_words))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


async def _tick_progress(attempt_id: int, progress_state: dict) -> None:
    try:
        while progress_state["value"] < _PROGRESS_TICK_CAP:
            await asyncio.sleep(_PROGRESS_TICK_SECONDS)
            # Реальний прогрес (on_progress у _process) міг уже піти вперед,
            # поки ми спали — тоді просто зупиняємось, аби не відкотити назад.
            if progress_state["value"] >= _PROGRESS_TICK_CAP:
                return
            progress_state["value"] = min(_PROGRESS_TICK_CAP, progress_state["value"] + _PROGRESS_TICK_STEP)
            java_client.report_stress_progress(attempt_id, status="PROCESSING", progress=progress_state["value"])
    except asyncio.CancelledError:
        pass


async def _process(attempt_id: int, session_id: str, audio_path: Path, reference_words: list[str]) -> None:
    logger.info(f"Stress-check started for attempt {attempt_id} (session {session_id})")
    progress_state = {"value": 5}
    java_client.report_stress_progress(attempt_id, status="PROCESSING", progress=progress_state["value"])

    def on_progress(stage: str, percent: int) -> None:
        progress_state["value"] = percent
        java_client.report_stress_progress(attempt_id, status="PROCESSING", progress=percent)

    ticker_task = asyncio.create_task(_tick_progress(attempt_id, progress_state))
    try:
        result = await asyncio.to_thread(
            check_stress_in_recording, session_id, audio_path, reference_words, on_progress
        )
        status = "DONE" if result.status == "done" else "NOT_APPLICABLE"
        java_client.report_stress_result(attempt_id, status=status, result=result)
        logger.info(
            f"Stress-check finished for attempt {attempt_id}: status={status}, "
            f"accuracy={result.accuracy}, checked={result.checked_words}"
        )
    except StressCheckError as e:
        logger.error(f"Stress-check failed for attempt {attempt_id}: {e}")
        java_client.report_stress_result(attempt_id, status="FAILED", error=str(e))
    except Exception as e:
        logger.exception(f"Stress-check crashed for attempt {attempt_id}")
        java_client.report_stress_result(attempt_id, status="FAILED", error=f"Unexpected error: {e}")
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task
