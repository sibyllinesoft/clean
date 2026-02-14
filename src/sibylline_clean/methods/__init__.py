"""Detection method registry.

Methods are registered at import time and looked up by name.
Third parties can register custom methods via register_method().
"""

from .base import DetectionMethod, MethodResult

_REGISTRY: dict[str, type[DetectionMethod]] = {}


def register_method(cls: type[DetectionMethod]) -> type[DetectionMethod]:
    """Register a detection method class. Can be used as a decorator."""
    _REGISTRY[cls.name()] = cls
    return cls


def get_method(name: str) -> type[DetectionMethod]:
    """Look up a registered detection method by name.

    Raises:
        ValueError: If the method name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown detection method {name!r}. Available methods: {available}")
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(_REGISTRY.keys())


# Register built-in methods
from .crf import SemiMarkovCRFMethod  # noqa: E402
from .heuristic import HeuristicMethod  # noqa: E402
from .promptshield import PromptShieldMethod  # noqa: E402

register_method(HeuristicMethod)
register_method(SemiMarkovCRFMethod)
register_method(PromptShieldMethod)

__all__ = [
    "DetectionMethod",
    "MethodResult",
    "register_method",
    "get_method",
    "list_methods",
    "HeuristicMethod",
    "SemiMarkovCRFMethod",
    "PromptShieldMethod",
]
