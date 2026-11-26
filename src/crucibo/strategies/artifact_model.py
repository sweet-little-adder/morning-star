"""Bar strategy backed by a morning-star artifact (mlp_v1 | linear_v1 | trees_v1).

Signal uses only market features of the bar close (position and cash are never model inputs for
linear_v1 / trees_v1; mlp_v1 artifacts exported by the comparison harness zero those weights).
With latency_bars=1 the runner fills at the next bar open.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from crucibo.aapl_mlp_v1.loader import load_artifact_model
from crucibo.core.bar import Bar
from crucibo.features import FeatureState
from crucibo.models import TradeTick
from crucibo.strategies.base import BarContext


class ArtifactModelStrategy:
    """sizing="fixed": target_shares from the artifact when long.
    sizing="equity": all equity at the signal-bar close on entry, then hold that share count
    until the signal turns flat (no per-bar rebalancing).
    """

    def __init__(
        self,
        *,
        model_path: Path,
        sizing: str = "fixed",
        active_from_ns: int | None = None,
    ) -> None:
        if sizing not in ("fixed", "equity"):
            raise ValueError("sizing must be 'fixed' or 'equity'")
        self._model = load_artifact_model(model_path)
        self._sizing = sizing
        self._active_from_ns = active_from_ns
        self._state = FeatureState(
            lookback=self._model.lookback,
            initial_cash=self._model.initial_cash,
            max_position=self._model.target_shares,
        )
        self._signals: list[bool] = []

    def prepare(self, bars: list[Bar]) -> None:
        self._state.reset()
        self._signals = []
        for bar in bars:
            tick = TradeTick(
                ts_event_ns=bar.ts_close_ns, symbol=bar.symbol, price=bar.close, size=1
            )
            vec = self._state.update(tick, position=0, cash=self._model.initial_cash)
            if vec is None:
                self._signals.append(False)
                continue
            self._signals.append(self._model.decide_shares(np.asarray(vec)) > 0)

    def on_bar(self, ctx: BarContext) -> int:
        if self._active_from_ns is not None and ctx.bar.ts_close_ns < self._active_from_ns:
            return 0
        if not self._signals[ctx.index]:
            return 0
        if self._sizing == "fixed":
            return int(self._model.target_shares)
        if ctx.portfolio.shares > 0:
            return ctx.portfolio.shares
        return int(math.floor(ctx.portfolio.equity / ctx.bar.close))
