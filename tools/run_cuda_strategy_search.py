from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _preflight(csv_path: Path, start: str, end: str) -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from numba import cuda
    import cupy as cp
    from src.v2.backtest_v2 import load_csv

    if not cuda.is_available() or len(cuda.gpus) == 0:
        raise SystemExit("CUDA is not available. Stop before CPU fallback.")

    device = cuda.get_current_device()
    cp.cuda.runtime.getDeviceCount()
    print(f"[preflight] CUDA OK: {device.name.decode()}")

    df = load_csv(str(csv_path))
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if len(str(end).strip()) == 10 and str(end).count("-") == 2:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    sample = df.loc[mask]
    if sample.empty:
        raise SystemExit(f"No rows in requested window: {start} to {end}")

    print(
        "[preflight] data OK: "
        f"rows={len(sample)}, start={sample['timestamp'].min()}, end={sample['timestamp'].max()}"
    )


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run the CUDA EMA strategy search in offline mode.")
    parser.add_argument(
        "--profile",
        default="smoke",
        choices=[
            "smoke",
            "phase1",
            "phase2",
            "phase3_long_only",
            "phase3_guard",
            "phase3_paper",
            "phase4_robust",
            "phase5_bear_defense",
            "phase6_htf_slope_defense",
            "phase7_pullback_slope",
            "phase8_trend_quality",
            "phase9_session_quality",
            "phase10_bear_chop_strict",
            "phase11_regime_ablation",
            "phase12_h4_slope_block",
            "phase13_rsi_quality",
            "phase14_hour_quality",
            "phase15_current_defense",
            "phase16_trend_breakout_regime",
            "phase17_long_breakout_regime",
            "phase18_htf_regime_breakout",
            "phase19_plateau_breakout_regime",
            "phase20_trailing_breakout_regime",
            "phase21_h1_atr_kill_breakout",
            "phase22_session_spread_breakout",
            "phase23_pullback_atr_regime",
            "phase24_high_conviction_breakout",
            "phase25_stale_exit_session_breakout",
            "phase26_bear_hedged_breakout",
            "phase27_tight_bidir_pf_breakout",
            "phase28_smooth_bidir_pullback",
            "phase29_risk_scaled_breakout_selector",
            "phase30_dsr_guarded_breakout",
            "phase31_weekday_quality_breakout",
            "phase32_trailing_bidir_quality",
            "phase33_funding_aware_breakout",
            "phase34_breakeven_quality_breakout",
            "phase35_vol_target_breakout",
            "phase36_drawdown_sized_breakout",
            "phase37_donchian_event_breakout",
            "phase38_donchian_weekday_veto",
            "phase39_donchian_h4_slope_block",
            "phase40_donchian_vol_target",
            "phase41_donchian_vol_return_recovery",
            "phase42_donchian_vol_guarded_recovery",
            "phase43_donchian_session_confirm",
            "phase44_donchian_funding_guard",
            "phase45_donchian_htf_atr_cap",
            "phase46_donchian_dd_taper",
            "phase47_donchian_exit_cooldown",
            "phase48_donchian_h4_micro_block",
            "phase49_donchian_time_stop_recovery",
            "phase50_donchian_smooth_frequency",
            "phase51_donchian_dense_low_rr",
            "phase52_donchian_chop_filter",
            "full",
        ],
    )
    parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-14")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0)
    parser.add_argument("--funding-csv", default=None, help="Optional Binance funding-rate CSV for actual funding event modeling.")
    parser.add_argument("--strict-min-return", type=float, default=30.0)
    parser.add_argument("--strict-min-pf", type=float, default=1.3)
    parser.add_argument("--strict-max-mdd", type=float, default=25.0)
    parser.add_argument("--strict-min-trades", type=int, default=30)
    parser.add_argument("--rank-metric", default="return", choices=["return", "strict_return", "robust", "smooth", "dense"])
    parser.add_argument("--param-start-index", type=int, default=0)
    parser.add_argument("--param-limit", type=int, default=0)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = _abs_path(root, args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    _preflight(csv_path, args.start, args.end)

    env = os.environ.copy()
    env.update(
        {
            "OFFLINE_OHLCV_H1": str(csv_path),
            "SEARCH_PROFILE": args.profile,
            "BT_START_DATE": args.start,
            "BT_END_DATE": args.end,
            "RESULTS_TOP_K": str(args.top_k),
            "CUDA_BATCH_SIZE": str(args.batch_size),
            "COMMISSION_RATE": str(args.commission),
            "SLIPPAGE_RATE": str(args.slippage),
            "ENTRY_DELAY_BARS": str(args.entry_delay_bars),
            "FUNDING_RATE_PER_8H": str(args.funding_rate_per_8h),
            "FUNDING_RATE_CSV": str(_abs_path(root, args.funding_csv)) if args.funding_csv else "",
            "STRICT_MIN_RETURN_PCT": str(args.strict_min_return),
            "STRICT_MIN_PF": str(args.strict_min_pf),
            "STRICT_MAX_MDD_PCT": str(args.strict_max_mdd),
            "STRICT_MIN_TRADES": str(args.strict_min_trades),
            "RESULTS_RANK_METRIC": args.rank_metric,
            "PARAM_START_INDEX": str(args.param_start_index),
            "PARAM_LIMIT": str(args.param_limit),
        }
    )

    cmd = [args.python, str(root / "emacrossmart.py")]
    print("[run] " + " ".join(cmd))
    print(
        "[run] env "
        f"SEARCH_PROFILE={args.profile} BT_START_DATE={args.start} BT_END_DATE={args.end} "
        f"CUDA_BATCH_SIZE={args.batch_size} RESULTS_TOP_K={args.top_k} "
        f"COMMISSION_RATE={args.commission} SLIPPAGE_RATE={args.slippage} "
        f"ENTRY_DELAY_BARS={args.entry_delay_bars} FUNDING_RATE_PER_8H={args.funding_rate_per_8h} "
        f"FUNDING_RATE_CSV={args.funding_csv or 'none'} "
        f"STRICT={args.strict_min_return}/{args.strict_min_pf}/{args.strict_max_mdd}/{args.strict_min_trades} "
        f"RESULTS_RANK_METRIC={args.rank_metric} PARAM_SLICE={args.param_start_index}:{args.param_limit or 'end'}"
    )
    if args.dry_run:
        return 0

    timeout = None if args.timeout_minutes <= 0 else args.timeout_minutes * 60
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=root, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[run] timed out after {args.timeout_minutes} minutes")
        return 124
    finally:
        print(f"[run] elapsed_seconds={time.time() - started:.1f}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
