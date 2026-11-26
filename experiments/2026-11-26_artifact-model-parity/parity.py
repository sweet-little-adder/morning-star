"""Parity: morning-star harness (next-open, equity sizing) vs crucibo run_backtest
(latency_bars=1, ArtifactModelStrategy) for exported logistic / linear / XGBoost / MLP.

Needs: pip install -e ../morning-star[compare], plus daily bars from
morning-star/scripts/fetch_yahoo_daily.py.
Usage: python parity.py --morning-star ../morning-star \
    --data-dir ../morning-star/data --out parity.json
Market data is read locally and never written into this repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")


def ts_close(d: str) -> int:
    t = dt.datetime.strptime(d, "%Y-%m-%d").replace(hour=16, tzinfo=NY)
    return int(t.timestamp() * 1e9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morning-star", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("parity.json"))
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--symbols", default="AAPL,GDDY,CCJ,SPY")
    a = ap.parse_args()
    sys.path.insert(0, str(a.morning_star / "src"))
    from morning_star.compare import fit as F
    from morning_star.compare.harness import features, fold_slices, labels, load, simulate
    from morning_star.export.artifact import save_artifact
    from morning_star.export.artifact_models import save_linear_artifact, save_trees_artifact

    from crucibo.backtest.economics import EconomicsConfig
    from crucibo.backtest.runner import run_backtest
    from crucibo.core.bar import Bar
    from crucibo.core.types import Vendor
    from crucibo.strategies.registry import resolve_strategy

    cfgj = json.loads((a.morning_star / "configs" / "compare_v2.json").read_text())
    mc = cfgj["models"]
    FEE, SLIP = 4.0, 2.0
    out = {"test_year": a.test_year, "fee_bps": FEE, "slippage_bps": SLIP, "rows": []}
    for sym in a.symbols.split(","):
        df = load(sym, str(a.data_dir))
        c, o = df.c.values, df.o.values
        X, fwd = features(c), labels(c)
        valid = ~np.isnan(X[:, 0]) & ~np.isnan(fwd)
        y = (fwd > 0).astype(float)
        (_, tr, te), = [f for f in fold_slices(df.date.values, [a.test_year], 500, valid)]
        models = {
            "logistic": F.fit_logistic(X[tr], y[tr], target_shares=1),
            "linear_reg": F.fit_linear(X[tr], fwd[tr], target_shares=1),
        }
        _, ens = F.fit_xgboost(X[tr], y[tr], mc["xgboost"], target_shares=1)
        models["xgboost"] = ens
        sc, P = F.fit_mlp_v2(X[tr], y[tr], mc["mlp_v2"])
        mlp = F.mlp_v2_to_v1(sc, P, target_shares=1)
        first = int(te[0])
        warm = max(0, first - 30)
        idxs = list(range(warm, len(df)))
        bars = [
            Bar(ts_close_ns=ts_close(df.date[i]), symbol=sym, interval="daily",
                vendor=Vendor.ALPHAVANTAGE, open=float(o[i]), high=float(df.h[i]),
                low=float(df.l[i]), close=float(c[i]), volume=float(df.v[i]))
            for i in idxs
        ]
        start_ns = ts_close(df.date[first])
        for name in ["logistic", "linear_reg", "xgboost", "mlp_v2"]:
            with tempfile.TemporaryDirectory() as td:
                d = Path(td) / "art"
                if name == "mlp_v2":
                    save_artifact(mlp, d, ticks_path=Path("n/a"), tick_count=len(tr), symbol=sym)
                    score = F.mlp_v2_proba(sc, P, X[te])
                    thr = 0.5
                elif name == "xgboost":
                    save_trees_artifact(ens, d)
                    score = None
                else:
                    save_linear_artifact(models[name], d)
                    m = models[name]
                    z = ((X[te] - m.mu) / m.sd) @ m.w + m.b
                    score = z if m.link == "identity" else 1 / (1 + np.exp(-z))
                    thr = m.threshold
                if name == "xgboost":
                    import xgboost as xgb  # noqa: F401 (reference only)

                    _, ens_ref = None, ens
                    score = np.array([ens_ref.predict_score(np.r_[X[i], 0.0, 1.0]) for i in te])
                    thr = 0.5
                pos = (score >= thr).astype(float)
                h_ret, _, h_pos = simulate(o, c, pos, te, FEE + SLIP)
                h_total = float(np.prod(1 + h_ret) - 1)
                strat = resolve_strategy("artifact", model_path=d, sizing="equity",
                                         active_from_ns=start_ns)
                cfg = EconomicsConfig(initial_cash=1e6, fee_bps=FEE, slippage_bps=SLIP,
                                      latency_bars=1, clamp_buys_to_cash=True)
                res = run_backtest(strategy=strat, bars=bars, cfg=cfg)
                eq = [e["equity_marked"] for e in res.equity_curve]
                i0 = [k for k, b in enumerate(bars) if b.ts_close_ns == start_ns][0]
                start_eq = eq[i0 - 1] if i0 > 0 else 1e6
                end_eq = eq[-1] * (1 - (FEE + SLIP) / 1e4 if res.final_shares else 1.0)
                c_total = end_eq / start_eq - 1
                # decision parity: fill bars == day after each position change in harness
                want = []
                prev = 0.0
                for j, t in enumerate(te):
                    if h_pos[j] != prev and t + 1 < len(df):
                        want.append(ts_close(df.date[t + 1]))
                    prev = h_pos[j]
                got = [f["ts_event_ns"] for f in res.fills]
                out["rows"].append(dict(
                    symbol=sym, model=name, n_fills=len(got),
                    fill_dates_match=(want == got),
                    harness_total_return=h_total, crucibo_total_return=float(c_total),
                    abs_diff=float(abs(h_total - c_total)),
                ))
                print(out["rows"][-1])
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
