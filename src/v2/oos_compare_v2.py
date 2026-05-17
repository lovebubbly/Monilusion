# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .backtest_v2 import load_csv, simulate


OUT_DIR = os.path.join(os.getcwd(), "wfa_optimized_params_output")
SYMBOL = "BTCUSDT"
TF_TAG = "1h"


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def to_is_performance(rpt: Dict) -> Dict:
    # Map from rpt keys to oos_compare expected fields
    return {
        "initial_balance": rpt.get("Initial Balance"),
        "final_balance": rpt.get("Final Balance"),
        "total_net_pnl": rpt.get("Total Net Pnl"),
        "total_net_pnl_percentage": rpt.get("Total Net Pnl Percentage"),
        "num_trades": rpt.get("Num Trades"),
        "num_wins": rpt.get("Num Wins"),
        "num_losses": rpt.get("Num Losses"),
        "win_rate_percentage": rpt.get("Win Rate Percentage"),
        "profit_factor": rpt.get("Profit Factor"),
        "max_drawdown_percentage": rpt.get("Max Drawdown Percentage"),
    }


def save_json(tag: str, is_perf: Dict, meta: Dict) -> str:
    ensure_outdir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"optimized_params_{SYMBOL}_{TF_TAG}_{ts}_{tag}.json"
    fpath = os.path.join(OUT_DIR, fname)
    payload = {
        "tag": tag,
        "symbol": SYMBOL,
        "timeframe": TF_TAG,
        "is_performance": is_perf,
        "meta": meta,
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[oos_compare_v2] Saved: {fpath}")
    return fpath


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    return df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)


def run_pair(csv_path: str, start: str, end: str, preset: str):
    df = load_csv(csv_path)
    dfp = filter_period(df, start, end)
    if len(dfp) == 0:
        raise SystemExit(f"No data in period {start}~{end}")

    # Baseline (real_M1 감성) — 게이트 off, 시간대 제한, 롱 온리, 고정 RR/SL
    base_params = {
        "ema_short": 19, "ema_long": 20, "rsi_n": 24,
        "adx_n": 14, "atr_n": 21,
        "sl_k": 2.8, "rr": 2.8, "trail_k": 2.5, "time_stop": 48,
        "allow_hours": [1, 3, 5, 8],
        "allow_shorts": False,
    }
    base_rpt, _ = simulate(dfp, base_params, initial_balance=10_000.0, risk_pct=0.02, fee_bps=2.0, no_gate=True, diag=False)
    base_perf = to_is_performance(base_rpt)
    save_json(f"baseline_{preset}", base_perf, {"start": start, "end": end, "params": base_params})

    # Improved (EV+) — HTF on, 게이트 on, 시간대 all, 롱 온리, BE/Trail, 완만한 풀백, RSI 완화
    imp_params = {
        "ema_short": 14, "ema_long": 100, "rsi_n": 21,
        "adx_n": 14, "atr_n": 21,
        "sl_k": 2.6, "rr": 3.5, "trail_k": 0.8, "time_stop": 36,
        "min_adx": 12.0, "min_atr_pct": 0.0010,
        "long_rsi_low": 50.0, "long_rsi_high": 100.0,
        "allow_hours": list(range(24)),
        "cb_max_losses": 4, "cb_cool_bars": 48,
        "use_htf": True, "htf": "4h", "htf_ema_short": 14, "htf_ema_long": 100,
        "htf_adx_n": 14, "htf_min_adx": 10.0,
        "require_adx_rising": False, "use_rsi_cross": False,
        "entry_pullback_k": 0.4,
        "be_after_r": 0.8,
        "allow_shorts": False,
    }
    imp_rpt, _ = simulate(dfp, imp_params, initial_balance=10_000.0, risk_pct=0.02, fee_bps=2.0, no_gate=False, diag=False)
    imp_perf = to_is_performance(imp_rpt)
    save_json(f"improved_{preset}", imp_perf, {"start": start, "end": end, "params": imp_params})


def main():
    ap = argparse.ArgumentParser(description="OOS compare for v2 backtester (baseline vs improved)")
    ap.add_argument("--csv", type=str, default=os.path.join("data", "BTCUSDT_1h.csv"))
    ap.add_argument("--presets", nargs="*", default=["recent", "stress_july"])
    args = ap.parse_args()

    presets = {
        "recent": ("2025-05-01", "2025-08-13"),
        "stress_july": ("2025-07-01", "2025-07-31"),
    }

    for p in args.presets:
        if p not in presets:
            raise SystemExit(f"Unknown preset: {p}")
        start, end = presets[p]
        run_pair(args.csv, start, end, p)
    print("\n[oos_compare_v2] Done.")


if __name__ == "__main__":
    main()
