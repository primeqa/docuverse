"""Helpers for delegating subcommands to legacy ``scripts/*`` entry points.

Most legacy scripts read ``sys.argv`` directly inside their ``main()``. We
swap ``sys.argv`` for the duration of the call rather than refactor each
script — that's an explicit "out of scope" item in the streamlining plan.
"""
from __future__ import annotations

import importlib
import sys
from typing import Sequence


def run_main_with_argv(
    module_name: str,
    func_name: str,
    argv: Sequence[str] | None,
    *,
    argv0: str = "docuverse",
) -> int:
    """Import ``module_name`` and call ``func_name`` with a swapped ``sys.argv``.

    The function is invoked with no arguments (legacy ``main()`` signature).
    If it returns an int we propagate it as the exit code; otherwise 0.
    """
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    saved = sys.argv
    sys.argv = [argv0, *(argv or [])]
    try:
        rc = fn()
    finally:
        sys.argv = saved
    if rc is None:
        return 0
    try:
        return int(rc)
    except (TypeError, ValueError):
        return 0
