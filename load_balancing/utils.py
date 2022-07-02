import time
from contextlib import contextmanager
from typing import Callable, Generator
import psutil


@contextmanager
def timer() -> Generator[Callable[[], float], None, None]:
    """Performance timer context manager
    :yield: runtime: Method that returns the runtime of the function
    :rtype: Callable[[], float]
    """
    start = time.perf_counter()
    runtime = 0
    yield lambda: runtime
    runtime = time.perf_counter() - start


def available_cores():
    return psutil.cpu_count(logical=False) or 4
