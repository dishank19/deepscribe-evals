"""Registry that exposes available evaluation metrics."""

from __future__ import annotations

from typing import Callable, Dict

MetricFn = Callable[[str, dict], dict]

_REGISTRY: Dict[str, MetricFn] = {}


def register(name: str, fn: MetricFn) -> None:
    """Register a new metric under the given name."""

    if name in _REGISTRY:
        raise KeyError(f"Metric '{name}' already registered")
    _REGISTRY[name] = fn


def get_metrics() -> Dict[str, MetricFn]:
    """Return a copy of registered metrics."""

    return dict(_REGISTRY)


def get_metric(name: str) -> MetricFn:
    """Fetch a single metric callable."""

    return _REGISTRY[name]
