"""
Text-to-speech engines behind one interface. Engines are imported lazily, so a missing optional
dependency only disables the engine that needs it.
"""
from importlib import import_module
from typing import Dict, List, Tuple

from engines.base import EngineUnavailable, Voice, VoiceEngine

# name -> (module, class)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "xtts": ("engines.xtts", "XttsEngine"),
    "classic": ("engines.classic", "ClassicEngine"),
}

DEFAULT_ENGINE = "classic"


def engine_class(name: str):
    if name not in _REGISTRY:
        raise KeyError("Unknown engine '%s'. Available: %s" % (name, ", ".join(_REGISTRY)))
    module, cls = _REGISTRY[name]
    return getattr(import_module(module), cls)


def list_engines() -> List[Tuple[str, str, bool, str]]:
    """(name, display name, available, reason if unavailable) for every registered engine."""
    rows = []
    for name in _REGISTRY:
        cls = engine_class(name)
        ok, reason = cls.availability()
        rows.append((name, cls.display_name, ok, reason))
    return rows


def create_engine(name: str, **kwargs) -> VoiceEngine:
    return engine_class(name)(**kwargs)
