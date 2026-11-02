"""Common infrastructure for the heavy ML engines.

Engines are loaded lazily and only once: the UI starts instantly and a
missing optional dependency disables one feature instead of crashing the
whole tool.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class EngineUnavailable(RuntimeError):
    """Raised when an engine cannot run (missing package, model, key...)."""


@dataclass
class EngineStatus:
    name: str
    installed: bool
    loaded: bool
    detail: str


def modules_available(*modules: str) -> tuple[bool, str]:
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        return False, "pacchetti mancanti: " + ", ".join(missing)
    return True, "installato"


class LazyEngine(Generic[T]):
    """Thread-safe, keyed lazy singleton around an expensive loader."""

    def __init__(self, name: str, modules: tuple[str, ...], loader: Callable[..., T]):
        self.name = name
        self.modules = modules
        self._loader = loader
        self._instances: dict[tuple, T] = {}
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        return modules_available(*self.modules)

    def get(self, *key) -> T:
        if key in self._instances:
            return self._instances[key]
        ok, reason = self.available()
        if not ok:
            raise EngineUnavailable(f"{self.name}: {reason}")
        with self._lock:
            if key not in self._instances:
                log.info("Caricamento motore %s %s...", self.name, key or "")
                try:
                    self._instances[key] = self._loader(*key)
                except EngineUnavailable:
                    raise
                except Exception as exc:  # pragma: no cover - depends on env
                    raise EngineUnavailable(f"{self.name}: impossibile caricare il modello ({exc})") from exc
        return self._instances[key]

    def status(self) -> EngineStatus:
        ok, detail = self.available()
        return EngineStatus(self.name, ok, bool(self._instances), detail)
