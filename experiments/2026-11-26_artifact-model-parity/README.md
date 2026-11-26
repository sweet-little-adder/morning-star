# Artifact model parity (2026-11-26)

**Goal:** check that logistic, linear, XGBoost and standardised-MLP artifacts exported by
morning-star (`feat/model-comparison-harness`) run through crucibo's bar backtest
(`latency_bars=1`, `ArtifactModelStrategy`, equity sizing, 4 bps fee + 2 bps slippage) and agree
with the morning-star harness.

```bash
pip install -e ../morning-star[compare]
python ../morning-star/scripts/fetch_yahoo_daily.py     # local data/, not committed
python parity.py --morning-star ../morning-star --data-dir data --out parity.json
```

**Result (`parity_2026_results.json`, test year 2026, AAPL/GDDY/CCJ/SPY x 4 models):**

- Fill dates match the harness decisions (next-open) in 16 of 16 runs.
- Total return differs by at most 0.0066 percentage points (6.5e-5 as a fraction) in 16 of 16 runs.
- Both sides use one convention: shares sized at the signal-bar close, filled at the next open,
  buys clamped to cash (`EconomicsConfig(clamp_buys_to_cash=True)`), fee 4 + slippage 2 bps. An
  earlier harness version assumed 100% of equity at the open, which made GDDY logistic differ by 8.9
  points after a -22.7% overnight gap; the harness was changed, not excepted.

Not covered: the `crucibo backtest` CLI. On this checkout `crucibo.data` is missing from git
(`data/` in `.gitignore` also matches `src/crucibo/data`), so the CLI cannot be imported. Parity
here uses `run_backtest` directly.
