"""linear_v1 / trees_v1 / mlp_v1(from standardised MLP) artifacts: scores match the original fits,
v1 compatibility is preserved, and the bar backtest path (next-open fills) uses them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from crucibo.aapl_mlp_v1.loader import load_aapl_mlp_v1_model, load_artifact_model
from crucibo.backtest.economics import EconomicsConfig
from crucibo.backtest.runner import run_backtest
from crucibo.core.bar import Bar
from crucibo.core.types import Vendor
from crucibo.strategies.registry import resolve_strategy

FIX = Path(__file__).parent / "fixtures" / "artifacts"
EXP = json.loads((FIX / "expected.json").read_text())
ROWS = np.array(EXP["rows"])


def _scores(name: str) -> np.ndarray:
    m = load_artifact_model(FIX / name)
    fn = m.predict_proba if hasattr(m, "predict_proba") else m.predict_score
    return np.array([fn(r) for r in ROWS])


def test_logistic_matches_reference_fit() -> None:
    np.testing.assert_allclose(_scores("linear_v1_logistic"), EXP["logistic"], atol=1e-9)


def test_linear_regression_matches_reference_fit() -> None:
    np.testing.assert_allclose(_scores("linear_v1_regression"), EXP["linear"], atol=1e-9)


def test_xgboost_trees_match_reference_fit() -> None:
    np.testing.assert_allclose(_scores("trees_v1_xgboost"), EXP["xgboost"], atol=1e-5)


def test_standardised_mlp_exports_as_plain_mlp_v1() -> None:
    m = load_aapl_mlp_v1_model(FIX / "mlp_v2_as_v1")  # strict legacy loader accepts it
    got = np.array([m.predict_proba(r) for r in ROWS])
    np.testing.assert_allclose(got, EXP["mlp_v2"], atol=1e-9)
    assert np.all(m.w1[3:] == 0)  # position_norm / cash_norm cannot influence decisions


def test_position_and_cash_do_not_change_new_model_scores() -> None:
    for name in ("linear_v1_logistic", "trees_v1_xgboost", "mlp_v2_as_v1"):
        m = load_artifact_model(FIX / name)
        fn = m.predict_proba if hasattr(m, "predict_proba") else m.predict_score
        r = ROWS[3].copy()
        r2 = r.copy()
        r2[3], r2[4] = 1.0, 0.1
        assert abs(fn(r) - fn(r2)) < 1e-12, name


def test_legacy_strict_loader_still_rejects_new_types() -> None:
    with pytest.raises(ValueError, match="unsupported aapl_mlp_v1 model_type"):
        load_aapl_mlp_v1_model(FIX / "trees_v1_xgboost")


def test_unknown_model_type_rejected(tmp_path: Path) -> None:
    d = tmp_path / "a"
    shutil.copytree(FIX / "linear_v1_logistic", d)
    mf = json.loads((d / "manifest.json").read_text())
    mf["model_type"] = "mystery_v9"
    (d / "manifest.json").write_text(json.dumps(mf))
    with pytest.raises(ValueError, match="unsupported artifact model_type"):
        load_artifact_model(d)


def test_bad_feature_indices_rejected(tmp_path: Path) -> None:
    d = tmp_path / "a"
    shutil.copytree(FIX / "linear_v1_logistic", d)
    mf = json.loads((d / "manifest.json").read_text())
    mf["feature_indices"] = [0, 1, 9]
    (d / "manifest.json").write_text(json.dumps(mf))
    with pytest.raises(ValueError, match="feature_indices"):
        load_artifact_model(d)


def _bars(n: int = 120) -> list[Bar]:
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    out = []
    for i in range(n):
        hi = max(open_[i], close[i]) * 1.002
        lo = min(open_[i], close[i]) * 0.998
        out.append(
            Bar(
                ts_close_ns=1_700_000_000_000_000_000 + i * 86_400_000_000_000,
                symbol="SYN", interval="daily", vendor=Vendor.ALPHAVANTAGE,
                open=float(open_[i]), high=float(hi), low=float(lo), close=float(close[i]),
                volume=1e6,
            )
        )
    return out


@pytest.mark.parametrize("name", ["linear_v1_logistic", "trees_v1_xgboost", "mlp_v2_as_v1"])
def test_artifact_strategy_fills_at_next_open(name: str) -> None:
    bars = _bars()
    strat = resolve_strategy("artifact", model_path=FIX / name, sizing="equity")
    cfg = EconomicsConfig(initial_cash=1e6, fee_bps=4, slippage_bps=2, latency_bars=1,
                          clamp_buys_to_cash=True)
    out = run_backtest(strategy=strat, bars=bars, cfg=cfg)
    by_ts = {b.ts_close_ns: b for b in bars}
    assert out.fills, "expected at least one trade on a 120-bar synthetic path"
    for f in out.fills:
        bar = by_ts[f["ts_event_ns"]]
        slip = 1 + (2e-4 if f["side"] == "BUY" else -2e-4)
        assert f["exec_price"] == pytest.approx(bar.open * slip)  # next-bar open, not signal close
        assert f["signal_bar_index"] < bars.index(bar)
    assert all(e["cash"] >= -1e-6 for e in out.equity_curve)


def test_equity_sizing_holds_shares_without_rebalancing() -> None:
    bars = _bars()
    strat = resolve_strategy("artifact", model_path=FIX / "linear_v1_logistic", sizing="equity")
    cfg = EconomicsConfig(initial_cash=1e6, latency_bars=1, clamp_buys_to_cash=True)
    out = run_backtest(strategy=strat, bars=bars, cfg=cfg)
    sides = [f["side"] for f in out.fills]
    # strictly alternating BUY/SELL
    assert all(a != b for a, b in zip(sides, sides[1:], strict=False))


def test_clamp_buys_to_cash_avoids_insufficient_cash_error() -> None:
    bars = _bars(40)
    strat = resolve_strategy("artifact", model_path=FIX / "linear_v1_logistic", sizing="fixed")
    cfg = EconomicsConfig(initial_cash=50.0, latency_bars=1)
    with pytest.raises(ValueError, match="insufficient cash"):
        run_backtest(strategy=strat, bars=bars, cfg=cfg)
    cfg2 = EconomicsConfig(initial_cash=50.0, latency_bars=1, clamp_buys_to_cash=True)
    run_backtest(strategy=resolve_strategy(
        "artifact", model_path=FIX / "linear_v1_logistic", sizing="fixed"), bars=bars, cfg=cfg2)


def test_artifact_strategy_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        resolve_strategy("artifact")
