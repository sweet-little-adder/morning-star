"""linear_v1 and trees_v1 artifact models (read-only evaluators; no sklearn/xgboost needed)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _market(features: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.asarray(features, dtype=np.float64).reshape(-1)[list(indices)]


@dataclass(frozen=True)
class LinearV1Model:
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    link: str  # "sigmoid" | "identity"
    feature_indices: tuple[int, ...]
    lookback: int
    forward_horizon: int
    threshold: float
    target_shares: int
    initial_cash: float

    def predict_score(self, features: np.ndarray) -> float:
        x = _market(features, self.feature_indices)
        z = float(((x - self.mu) / self.sd) @ self.w + self.b)
        if self.link == "identity":
            return z
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def decide_shares(self, features: np.ndarray) -> int:
        return self.target_shares if self.predict_score(features) >= self.threshold else 0


@dataclass(frozen=True)
class TreesV1Model:
    """XGBoost-style ensemble: left when float32(x[f]) < float32(split); NaN follows default."""

    tree_offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    split: np.ndarray
    value: np.ndarray
    default_left: np.ndarray
    base_margin: float
    feature_indices: tuple[int, ...]
    lookback: int
    forward_horizon: int
    threshold: float
    target_shares: int
    initial_cash: float

    @property
    def n_trees(self) -> int:
        return int(len(self.tree_offsets) - 1)

    def predict_margin(self, features: np.ndarray) -> float:
        x32 = _market(features, self.feature_indices).astype(np.float32)
        total = self.base_margin
        for t in range(self.n_trees):
            n = int(self.tree_offsets[t])
            while self.left[n] >= 0:
                v = float(x32[self.feature[n]])
                if math.isnan(v):
                    n = int(self.left[n] if self.default_left[n] else self.right[n])
                elif v < float(np.float32(self.split[n])):
                    n = int(self.left[n])
                else:
                    n = int(self.right[n])
            total += float(self.value[n])
        return total

    def predict_score(self, features: np.ndarray) -> float:
        m = max(-60.0, min(60.0, self.predict_margin(features)))
        return 1.0 / (1.0 + math.exp(-m))

    def decide_shares(self, features: np.ndarray) -> int:
        return self.target_shares if self.predict_score(features) >= self.threshold else 0
