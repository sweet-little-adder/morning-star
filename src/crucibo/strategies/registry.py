"""Named strategy factory for CLI and programmatic backtests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from crucibo.strategies.artifact_model import ArtifactModelStrategy
from crucibo.strategies.base import Strategy
from crucibo.strategies.ma_crossover import MaCrossoverStrategy
from crucibo.strategies.mean_reversion import MeanReversionStrategy

StrategyFactory = Callable[..., Strategy]

def _artifact(**kw: Any) -> Strategy:
    model_path = kw.get("model_path")
    if model_path is None:
        raise ValueError("artifact strategy requires model_path (artifact directory)")
    return ArtifactModelStrategy(
        model_path=Path(model_path),
        sizing=str(kw.get("sizing", "fixed")),
        active_from_ns=kw.get("active_from_ns"),
    )


_REGISTRY: dict[str, StrategyFactory] = {
    "artifact": _artifact,
    "ma_crossover": lambda **kw: MaCrossoverStrategy(
        fast=int(kw.get("fast", 12)),
        slow=int(kw.get("slow", 26)),
        target_shares=int(kw.get("target_shares", 100)),
    ),
    "mean_reversion": lambda **kw: MeanReversionStrategy(
        lookback=int(kw.get("lookback", 20)),
        entry_z=float(kw.get("entry_z", 1.0)),
        target_shares=int(kw.get("target_shares", 100)),
    ),
}


def list_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_strategy(name: str, **kwargs: Any) -> Strategy:
    key = name.strip().lower().replace("-", "_")
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"unknown strategy {name!r} (try {' | '.join(list_strategies())})")
    return factory(**kwargs)