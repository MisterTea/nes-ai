import asyncio
import inspect
import logging
import multiprocessing
import os
import tracemalloc
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

from pyinstrument import Profiler
from pyinstrument.session import Session

logger = logging.getLogger(__name__)

os.environ["PYINSTRUMENT_IGNORE_OVERHEAD_WARNING"] = (
    "1"  # If the cpu is overloaded, pyinstrument will throw a warning.  We ignore it here.
)


def get_main_pid():
    # Gets the pid of the main python process.

    p_process = multiprocessing.parent_process()
    if (
        multiprocessing.current_process().name == "MainProcess"
        or p_process is None
        or p_process.pid == 1
    ):
        retval = multiprocessing.current_process().pid
    else:
        assert p_process is not None, "Parent process should not be None"

        # If we run a multiprocessing pool inside a multiprocessing pool, this could fail.  Don't do that.
        assert (
            p_process.name == "MainProcess"
        ), f"Unexpected parent process name: {p_process.name}"

        retval = p_process.pid
    assert retval is not None, "Returned PID should not be None"
    return retval


def get_session_log_directory():
    pid = get_main_pid()
    session_log_dir = f"/tmp/pyinstrument/sessions/{pid}"
    os.makedirs(session_log_dir, exist_ok=True)
    return session_log_dir


class MemoryProfiler:
    def __init__(self):
        self.active = False

    def start(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.active = True

    def stop(self) -> str:
        if self.active:
            final, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("filename")
            tracemalloc.stop()
            top_file_info = "\n".join([str(t) for t in top_stats[:10]])
            return f"""Memory usage peak: {round(peak / 1024**2,2)} MB, final: {round(final / 1024**2,2)} MB, top files:
{top_file_info}"""
        return "Memory profiler is not active"


@contextmanager
def profile(
    print_session: bool = False, log_session: bool = True, profile_memory: bool = True
):
    profiler = Profiler(async_mode="enabled")
    profiler.start(caller_frame=inspect.currentframe())
    memory_profiler = MemoryProfiler()
    if profile_memory:
        memory_profiler.start()

    try:
        logger.info("Profiler started")
        yield
    finally:
        session = profiler.stop()
        profiler_text = profiler.output_text(unicode=True, color=True)
        if profile_memory:
            memory_usage_text = memory_profiler.stop()
            if print_session:
                logger.info(profiler_text + "\n\n" + memory_usage_text)
        else:
            if print_session:
                logger.info(profiler_text)
        profiler.reset()

        if log_session:
            session_log_directory = get_session_log_directory()

            session_file_path = os.path.join(session_log_directory, f"{uuid4()}.json")
            session.save(session_file_path)


async def my_async_function():
    await asyncio.sleep(0.1)


async def job():
    with profile():
        await my_async_function()


def collect_and_cleanup_profile_files() -> Session | None:
    session_log_directory = get_session_log_directory()
    session = None
    for root, _, files in os.walk(session_log_directory):
        for file in files:
            full_path = os.path.join(root, file)
            if session is None:
                session = Session.load(full_path)
            else:
                new_session = Session.load(full_path)
                session = Session.combine(session, new_session)
            os.unlink(full_path)
    return session


def profiler_report():
    session = collect_and_cleanup_profile_files()
    if session is not None:
        profiler = Profiler()

        # Inject the session into a new profiler so we can render it
        profiler._last_session = session  # pylint: disable=protected-access
        logger.info(profiler.output_text(unicode=True, color=True))


async def profiler_report_task(interval: int = 60 * 15):
    try:
        while True:
            profiler_report()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Send one last report before exiting
        profiler_report()
    except BaseException as e:
        logger.error("Error in profiler_report_task: %s", e)
        raise


@asynccontextmanager
async def profiler_report_periodically(interval: int = 60 * 15):
    task = asyncio.create_task(profiler_report_task(interval))
    try:
        yield
    finally:
        task.cancel()
        await task
