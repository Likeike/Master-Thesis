import functools
import time
from typing import Callable, Any


def timed_func(func: Callable) -> Any:
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        t1 = time.perf_counter(), time.process_time()
        res = func(*args, **kwargs)
        t2 = time.perf_counter(), time.process_time()
        print(f"{func.__name__}")
        print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
        print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
        return res

    return _wrapper
