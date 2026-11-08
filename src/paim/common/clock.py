"""A wall clock that AppWorld cannot freeze.

AppWorld freezes time inside the agent's process (freezegun) so its apps are deterministic: every `time.time()`
there returns the same instant, which made every measured latency and episode duration 0. The real functions are
captured at import, inside a dict, where freezegun's module-attribute patching does not reach them. Import this
module before AppWorld starts a task.
"""

from __future__ import annotations

import time as _time

_REAL = {"perf_counter": _time.perf_counter, "time": _time.time}


def perf_counter() -> float:
    return _REAL["perf_counter"]()


def wall_time() -> float:
    return _REAL["time"]()


__all__ = ["perf_counter", "wall_time"]
