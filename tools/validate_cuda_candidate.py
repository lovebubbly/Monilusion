from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from diff_cuda_cpu_reference import (
    _diff,
    _diff_pass,
    _intrabar_policy_comparison,
    _load_ohlcv,
    _num,
    cpu_reference_backtest,
    load_funding_cumulative,
)
from validate_v2_strategy import _deflated_sharpe, _monte_carlo


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": _num(metrics.get("total_net_pnl_percentage")),
        "net_pf": _num(metrics.get("net_profit_factor", metrics.get("profit_factor"))),
        "gross_pf": _num(metrics.get("gross_profit_factor")),
        "mdd_pct": _num(metrics.get("max_drawdown_percentage")),
        "trades": int(metrics.get("num_trades", 0)),
        "win_rate_pct": _num(metrics.get("win_rate_percentage")),
    }


def _score(metrics: dict[str, Any]) -> float:
    s = _summary(metrics)
    if s["trades"] <= 0:
        return -1e9
    return s["return_pct"] + 10.0 * math.log(max(0.01, s["net_pf"])) - 0.25 * s["mdd_pct"]


def _passes(metrics: dict[str, Any], args: argparse.Namespace, *, scope: str = "full") -> bool:
    s = _summary(metrics)
    if scope == "wfo":
        min_return = args.wfo_min_return
        min_pf = args.wfo_min_pf
        max_mdd = args.wfo_max_mdd
        min_trades = args.wfo_min_trades
    else:
        min_return = args.min_return
        min_pf = args.min_pf
        max_mdd = args.max_mdd
        min_trades = args.min_trades
    return (
        s["return_pct"] >= min_return
        and s["net_pf"] >= min_pf
        and s["mdd_pct"] <= max_mdd
        and s["trades"] >= min_trades
    )


def _gate(name: str, passed: bool, observed: Any, threshold: Any, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _candidate_gates(
    full_metrics: dict[str, Any],
    source_diff_pass: bool,
    wfo: dict[str, Any],
    pbo: dict[str, Any],
    dsr: dict[str, Any],
    mc: dict[str, Any],
    stress: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    s = _summary(full_metrics)
    pbo_enabled = bool(pbo.get("enabled"))
    pbo_value = _num(pbo.get("pbo")) if pbo_enabled else 1.0
    return [
        _gate(
            "source_period_cpu_gpu_diff",
            source_diff_pass,
            source_diff_pass,
            True,
            "CPU reference must reproduce saved CUDA result for the exact saved period.",
        ),
        _gate("full_return_pct", s["return_pct"] >= args.min_return, s["return_pct"], f">= {args.min_return}", "Expanded-period net return."),
        _gate("full_net_pf", s["net_pf"] >= args.min_pf, s["net_pf"], f">= {args.min_pf}", "Net PF after fees/slippage/funding."),
        _gate("full_mdd_pct", s["mdd_pct"] <= args.max_mdd, s["mdd_pct"], f"<= {args.max_mdd}", "Intrabar mark-to-market MDD."),
        _gate("full_trades", s["trades"] >= args.min_trades, s["trades"], f">= {args.min_trades}", "Minimum trade count."),
        _gate("wfo_pass_ratio", wfo["pass_ratio"] >= args.min_wfo_pass_ratio, wfo["pass_ratio"], f">= {args.min_wfo_pass_ratio}", "Rolling fixed-candidate WFO pass ratio."),
        _gate("pbo", pbo_enabled and pbo_value <= args.max_pbo, pbo.get("pbo", "disabled"), f"<= {args.max_pbo}", "Top-K purged/embargoed CPCV/PBO over saved CUDA candidates."),
        _gate("dsr", _num(dsr["dsr"]) >= args.min_dsr, dsr["dsr"], f">= {args.min_dsr}", "Deflated Sharpe Ratio."),
        _gate("mc_prob_return_positive", _num(mc.get("prob_return_positive", 0.0)) >= args.min_mc_prob_positive, mc.get("prob_return_positive", 0.0), f">= {args.min_mc_prob_positive}", "Monte Carlo bootstrap positive-return probability."),
        _gate("mc_return_p05", _num(mc.get("return_pct_p05", -float("inf"))) >= args.min_mc_return_p05, mc.get("return_pct_p05", -float("inf")), f">= {args.min_mc_return_p05}", "Monte Carlo 5th percentile return."),
        _gate("cost_stress_pass_ratio", stress["pass_ratio"] >= args.min_stress_pass_ratio, stress["pass_ratio"], f">= {args.min_stress_pass_ratio}", "Fee/slippage/funding stress matrix pass ratio."),
    ]


def _decision(gates: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [gate for gate in gates if not gate["pass"]]
    status = "PROMOTE_TO_SHADOW" if not failed else "HOLD_AUTOMATED_PAPER"
    return {
        "status": status,
        "ready_for_shadow": not failed,
        "ready_for_paper": False,
        "failed_gates": [gate["name"] for gate in failed],
        "rationale": (
            "All validation gates passed; candidate may enter shadow logging only."
            if not failed
            else "Candidate is not eligible for automated paper trading or shadow promotion until failed gates pass."
        ),
    }


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_parameters(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    canonical = dict(params)
    if not bool(canonical.get("use_htf_ema_filter", False)) and not bool(canonical.get("use_htf_slope_filter", False)):
        canonical.pop("ema_htf", None)
    if not bool(canonical.get("use_htf_slope_filter", False)):
        canonical.pop("htf_slope_lookback_bars", None)
        canonical.pop("htf_min_slope_pct", None)
        canonical.pop("htf_price_buffer_atr", None)
        canonical.pop("use_htf_slope_block_filter", None)
        canonical.pop("htf_block_slope_min_pct", None)
        canonical.pop("htf_block_slope_max_pct", None)
    else:
        block_min = float(canonical.get("htf_block_slope_min_pct", 0.0))
        block_max = float(canonical.get("htf_block_slope_max_pct", 0.0))
        if not bool(canonical.get("use_htf_slope_block_filter", False)) or block_max <= block_min:
            canonical.pop("use_htf_slope_block_filter", None)
            canonical.pop("htf_block_slope_min_pct", None)
            canonical.pop("htf_block_slope_max_pct", None)
    if not bool(canonical.get("use_htf_regime_filter", False)):
        canonical.pop("htf_adx_threshold", None)
        canonical.pop("htf_atr_percent_min", None)
        canonical.pop("htf_atr_percent_max", None)
    if not bool(canonical.get("use_h1_slope_filter", False)):
        canonical.pop("h1_slope_lookback_bars", None)
        canonical.pop("h1_min_slope_pct", None)
    if float(canonical.get("ema_spread_min_atr", 0.0)) <= 0.0:
        canonical.pop("ema_spread_min_atr", None)
    if not bool(canonical.get("use_h1_atr_percent_filter", False)):
        canonical.pop("h1_atr_percent_min", None)
        canonical.pop("h1_atr_percent_max", None)
    if canonical.get("entry_session_mode", "all") == "all":
        canonical.pop("entry_session_mode", None)
    if not bool(canonical.get("use_adx_filter", False)):
        canonical.pop("adx_period", None)
        canonical.pop("adx_threshold", None)
    if not bool(canonical.get("use_volume_filter", False)):
        canonical.pop("volume_sma_period", None)
    if not bool(canonical.get("use_rsi_filter", False)):
        canonical.pop("rsi_period", None)
        canonical.pop("rsi_threshold_long", None)
        canonical.pop("rsi_threshold_short", None)
    if not bool(canonical.get("use_regime_filter", False)):
        canonical.pop("adx_threshold_regime", None)
        canonical.pop("atr_percent_threshold_regime", None)
    entry_mode = canonical.get("entry_signal_mode", "crossover")
    if entry_mode in {"trend_breakout", "donchian_breakout"}:
        canonical.pop("adx_threshold_for_short", None)
        canonical.pop("rsi_momentum_threshold", None)
    elif not bool(canonical.get("allow_short_entries", True)):
        canonical.pop("adx_threshold_for_short", None)
        canonical.pop("price_breakdown_period", None)
        canonical.pop("rsi_momentum_threshold", None)
    if canonical.get("exit_strategy_type") == "FixedRR":
        canonical.pop("trailing_atr_period", None)
        canonical.pop("trailing_atr_multiplier", None)
        canonical.pop("profit_threshold_for_trail", None)
    else:
        canonical.pop("risk_reward_ratio", None)
    if float(canonical.get("drawdown_guard_pct", 0.0)) <= 0.0:
        canonical.pop("drawdown_guard_cooldown_bars", None)
    if not bool(canonical.get("use_breakeven_stop", False)):
        canonical.pop("breakeven_trigger_r", None)
        canonical.pop("breakeven_offset_r", None)
    if not bool(canonical.get("use_volatility_target_sizing", False)):
        canonical.pop("volatility_target_annual", None)
        canonical.pop("volatility_lookback_hours", None)
        canonical.pop("volatility_sizing_min_mult", None)
        canonical.pop("volatility_sizing_max_mult", None)
    if not bool(canonical.get("use_drawdown_sizing", False)):
        canonical.pop("drawdown_sizing_start_pct", None)
        canonical.pop("drawdown_sizing_full_pct", None)
        canonical.pop("drawdown_sizing_min_mult", None)
    return tuple(sorted(canonical.items()))


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    out = []
    for row in candidates:
        key = _canonical_parameters(row["parameters"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _effective_dsr_trials(obj: dict[str, Any], candidates: list[dict[str, Any]], override: int | None = None) -> tuple[int, str]:
    if override is not None and override > 0:
        return int(override), "manual_override"
    total_grid = int(obj.get("total_param_combinations") or 0)
    topk = len(candidates)
    if total_grid > 0:
        return max(2, total_grid, topk), "max(total_param_combinations, unique_topk_candidates)"
    return max(2, topk), "unique_topk_candidates"


def _select_rank(obj: dict[str, Any], rank: int) -> dict[str, Any]:
    for row in obj.get("results", []):
        if int(row["rank"]) == rank:
            return row
    raise SystemExit(f"No rank {rank} found.")


def _run(
    df: pd.DataFrame,
    params: dict[str, Any],
    initial: float,
    commission_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
    funding_rate_per_8h: float,
    include_equity: bool = False,
    intrabar_policy: str = "conservative",
    funding_cumulative: np.ndarray | None = None,
    funding_csv: Path | None = None,
):
    if funding_cumulative is None and funding_csv is not None:
        funding_cumulative = load_funding_cumulative(df, funding_csv)
    return cpu_reference_backtest(
        df,
        params,
        initial_balance=initial,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        entry_delay_bars=entry_delay_bars,
        funding_rate_per_8h=funding_rate_per_8h,
        include_equity=include_equity,
        intrabar_policy=intrabar_policy,
        funding_cumulative=funding_cumulative,
    )


def _intrabar_policy_band(
    df: pd.DataFrame,
    params: dict[str, Any],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    conservative_metrics: dict[str, Any] | None = None,
    funding_cumulative: np.ndarray | None = None,
    funding_csv: Path | None = None,
) -> dict[str, Any]:
    conservative = conservative_metrics
    if conservative is None:
        conservative = _run(
            df,
            params,
            args.initial,
            commission_rate,
            slippage_rate,
            args.entry_delay_bars,
            funding_rate_per_8h,
            intrabar_policy="conservative",
            funding_cumulative=funding_cumulative,
            funding_csv=funding_csv,
        )
    optimistic = _run(
        df,
        params,
        args.initial,
        commission_rate,
        slippage_rate,
        args.entry_delay_bars,
        funding_rate_per_8h,
        intrabar_policy="optimistic",
        funding_cumulative=funding_cumulative,
        funding_csv=funding_csv,
    )
    return _intrabar_policy_comparison(conservative, optimistic)


def _folds(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_bars: int,
    embargo_bars: int,
    min_rows: int,
) -> list[dict[str, Any]]:
    first = df.index.min().floor("h")
    last = df.index.max().ceil("h")
    out = []
    start = first
    while True:
        train_start = start
        train_end = start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last + pd.Timedelta(hours=1):
            break
        train = df[(df.index >= train_start) & (df.index < train_end)]
        test = df[(df.index >= train_end) & (df.index < test_end)]
        if purge_bars > 0 and len(train) > purge_bars:
            train = train.iloc[:-purge_bars]
        if embargo_bars > 0 and len(test) > embargo_bars:
            test = test.iloc[embargo_bars:]
        if len(train) >= min_rows and len(test) >= min_rows:
            out.append(
                {
                    "fold": len(out) + 1,
                    "train_start": train.index.min(),
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "test": test,
                }
            )
        start = start + pd.DateOffset(months=step_months)
    return out


def _default_wfo_min_trades(df: pd.DataFrame, min_trades: int, test_months: int) -> int:
    if df.empty:
        return 3
    total_days = max(1.0, (df.index.max() - df.index.min()).total_seconds() / 86400.0)
    total_months = max(1.0, total_days / 30.4375)
    return max(3, math.ceil(min_trades * test_months / total_months))


def _fixed_candidate_wfo(
    df: pd.DataFrame,
    params: dict[str, Any],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    funding_csv: Path | None = None,
) -> dict[str, Any]:
    rows = []
    pass_count = 0
    folds = _folds(
        df,
        args.train_months,
        args.test_months,
        args.step_months,
        args.purge_bars,
        args.embargo_bars,
        args.min_fold_rows,
    )
    for fold in folds:
        metrics = _run(
            fold["test"],
            params,
            args.initial,
            commission_rate,
            slippage_rate,
            args.entry_delay_bars,
            funding_rate_per_8h,
            funding_csv=funding_csv,
        )
        passed = _passes(metrics, args, scope="wfo")
        pass_count += int(passed)
        rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "summary": _summary(metrics),
                "pass": passed,
            }
        )
    return {"folds": rows, "num_folds": len(rows), "pass_ratio": pass_count / len(rows) if rows else 0.0}


def _topk_cscv_pbo(
    df: pd.DataFrame,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    funding_csv: Path | None = None,
) -> dict[str, Any]:
    splits = args.cscv_splits
    if splits < 4 or splits % 2 != 0 or len(candidates) < 2:
        return {"enabled": False, "reason": "requires even splits >=4 and at least two candidates"}
    block_edges = np.linspace(0, len(df), splits + 1, dtype=int)
    raw_blocks = [df.iloc[block_edges[i] : block_edges[i + 1]] for i in range(splits)]
    left_trim = max(0, int(getattr(args, "embargo_bars", 0)))
    right_trim = max(0, int(getattr(args, "purge_bars", 0)))
    blocks = []
    effective_rows = []
    for block in raw_blocks:
        if left_trim + right_trim >= len(block):
            trimmed = block.iloc[0:0]
        else:
            trimmed = block.iloc[left_trim : len(block) - right_trim]
        blocks.append(trimmed)
        effective_rows.append(int(len(trimmed)))
    scores = np.full((len(candidates), splits), -1e9, dtype=np.float64)
    for c_idx, row in enumerate(candidates):
        for b_idx, block in enumerate(blocks):
            if len(block) < args.min_fold_rows:
                continue
            metrics = _run(
                block,
                row["parameters"],
                args.initial,
                commission_rate,
                slippage_rate,
                args.entry_delay_bars,
                funding_rate_per_8h,
                funding_csv=funding_csv,
            )
            scores[c_idx, b_idx] = _score(metrics)

    lambdas = []
    ranks = []
    block_ids = range(splits)
    for train_blocks in itertools.combinations(block_ids, splits // 2):
        train_set = set(train_blocks)
        test_blocks = [idx for idx in block_ids if idx not in train_set]
        train_scores = np.nanmean(scores[:, list(train_blocks)], axis=1)
        test_scores = np.nanmean(scores[:, test_blocks], axis=1)
        if not np.isfinite(train_scores).any() or not np.isfinite(test_scores).any():
            continue
        winner = int(np.nanargmax(train_scores))
        order = np.argsort(np.argsort(test_scores))
        pct_rank = float(order[winner] / max(1, len(candidates) - 1))
        pct_rank = min(1.0 - 1e-9, max(1e-9, pct_rank))
        ranks.append(pct_rank)
        lambdas.append(math.log(pct_rank / (1.0 - pct_rank)))
    if not lambdas:
        return {"enabled": False, "reason": "no finite samples"}
    lambdas_arr = np.array(lambdas, dtype=np.float64)
    ranks_arr = np.array(ranks, dtype=np.float64)
    return {
        "enabled": True,
        "note": "Purged/embargoed CPCV/PBO over saved CUDA top-K rows only; each block score trims embargo bars from the left edge and purge bars from the right edge. Use larger top-K for stronger evidence.",
        "splits": splits,
        "candidates": len(candidates),
        "purge_bars": right_trim,
        "embargo_bars": left_trim,
        "raw_block_rows_min": int(min(len(block) for block in raw_blocks)) if raw_blocks else 0,
        "effective_block_rows_min": int(min(effective_rows)) if effective_rows else 0,
        "effective_block_rows": effective_rows,
        "samples": len(lambdas),
        "pbo": round(float(np.mean(lambdas_arr <= 0.0)), 4),
        "median_test_percentile": round(float(np.median(ranks_arr)), 4),
    }


def _cost_stress(
    df: pd.DataFrame,
    params: dict[str, Any],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    funding_cumulative: np.ndarray | None = None,
    funding_csv: Path | None = None,
) -> dict[str, Any]:
    rows = []
    pass_count = 0
    if funding_cumulative is None and funding_csv is not None:
        funding_cumulative = load_funding_cumulative(df, funding_csv)
    for fee_mult in [1.0, 1.5, 2.0]:
        for slip_mult in [1.0, 2.0]:
            for funding_mult in [0.0, 1.0, 2.0]:
                stressed_funding_cumulative = (
                    funding_cumulative * funding_mult if funding_cumulative is not None else None
                )
                metrics = _run(
                    df,
                    params,
                    args.initial,
                    commission_rate * fee_mult,
                    slippage_rate * slip_mult,
                    args.entry_delay_bars,
                    funding_rate_per_8h * funding_mult,
                    funding_cumulative=stressed_funding_cumulative,
                    funding_csv=funding_csv if stressed_funding_cumulative is None else None,
                )
                passed = _passes(metrics, args)
                pass_count += int(passed)
                rows.append(
                    {
                        "fee_multiplier": fee_mult,
                        "slippage_multiplier": slip_mult,
                        "funding_multiplier": funding_mult,
                        "summary": _summary(metrics),
                        "pass": passed,
                    }
                )
    return {"rows": rows, "pass_ratio": pass_count / len(rows) if rows else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a saved CUDA candidate with CPU reference WFO/stress checks.")
    parser.add_argument("--cuda-json", required=True, type=Path)
    parser.add_argument("--csv", default="data/BTCUSDT_1h.csv", type=Path)
    parser.add_argument("--start", default=None, help="Override validation start; defaults to CUDA JSON period_start.")
    parser.add_argument("--end", default=None, help="Override validation end; defaults to CUDA JSON period_end.")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--dsr-trials", type=int, default=None, help="Override DSR multiple-testing trial count; defaults to total_param_combinations when available.")
    parser.add_argument("--funding-csv", type=Path, default=None, help="Optional Binance funding-rate event CSV for actual funding charges.")
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--entry-delay-bars", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-fold-rows", type=int, default=300)
    parser.add_argument("--cscv-splits", type=int, default=4)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--wfo-min-return", type=float, default=0.0)
    parser.add_argument("--wfo-min-pf", type=float, default=1.0)
    parser.add_argument("--wfo-max-mdd", type=float, default=None)
    parser.add_argument("--wfo-min-trades", type=int, default=None)
    parser.add_argument("--min-wfo-pass-ratio", type=float, default=0.6)
    parser.add_argument("--max-pbo", type=float, default=0.5)
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--min-mc-prob-positive", type=float, default=0.6)
    parser.add_argument("--min-mc-return-p05", type=float, default=0.0)
    parser.add_argument("--min-stress-pass-ratio", type=float, default=0.5)
    parser.add_argument("--out", default="wfa_optimized_params_output/cuda_candidate_validation.json", type=Path)
    parser.add_argument("--fail-on-hold", action="store_true", help="Return exit code 2 when the decision is HOLD_AUTOMATED_PAPER.")
    args = parser.parse_args()
    if args.wfo_max_mdd is None:
        args.wfo_max_mdd = args.max_mdd

    obj = _load_results(args.cuda_json)
    selected = _select_rank(obj, args.rank)
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    period_start = args.start or obj["period_start"]
    period_end = args.end or obj["period_end"]
    df = _load_ohlcv(csv_path, period_start, period_end)
    if args.wfo_min_trades is None:
        args.wfo_min_trades = _default_wfo_min_trades(df, args.min_trades, args.test_months)
    source_df = _load_ohlcv(csv_path, obj["period_start"], obj["period_end"])
    commission_rate = float(obj.get("commission_rate", 0.0005))
    slippage_rate = float(obj.get("slippage_rate", 0.0002))
    funding_rate = float(obj.get("funding_rate_per_8h", 0.0))
    funding_csv = args.funding_csv
    if funding_csv is None and obj.get("funding_model") == "actual_funding_events" and obj.get("funding_rate_csv"):
        funding_csv = Path(obj["funding_rate_csv"])
    if funding_csv is not None and not funding_csv.is_absolute():
        funding_csv = ROOT / funding_csv
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    source_funding_cumulative = load_funding_cumulative(source_df, funding_csv) if funding_csv is not None else None
    entry_delay = int(obj.get("entry_delay_bars", 1)) if args.entry_delay_bars is None else args.entry_delay_bars
    args.entry_delay_bars = entry_delay

    full_metrics, equity = _run(
        df,
        selected["parameters"],
        args.initial,
        commission_rate,
        slippage_rate,
        entry_delay,
        funding_rate,
        include_equity=True,
        funding_cumulative=funding_cumulative,
    )
    source_metrics = _run(
        source_df,
        selected["parameters"],
        args.initial,
        commission_rate,
        slippage_rate,
        entry_delay,
        funding_rate,
        funding_cumulative=source_funding_cumulative,
    )
    source_diff = _diff(source_metrics, selected["performance"])
    source_diff_pass = _diff_pass(source_diff, 0.01, 0.01, 0.01, 0)
    intrabar_policy_band = _intrabar_policy_band(
        df,
        selected["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate,
        full_metrics,
        funding_cumulative=funding_cumulative,
    )
    wfo = _fixed_candidate_wfo(
        df,
        selected["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate,
        funding_csv=funding_csv,
    )
    raw_candidates = obj.get("results", [])[: args.max_candidates]
    candidates = _dedupe_candidates(raw_candidates)
    pbo = _topk_cscv_pbo(
        df,
        candidates,
        args,
        commission_rate,
        slippage_rate,
        funding_rate,
        funding_csv=funding_csv,
    )
    dsr_trials, dsr_trials_basis = _effective_dsr_trials(obj, candidates, args.dsr_trials)
    dsr = _deflated_sharpe(equity, dsr_trials)
    mc = _monte_carlo(equity, args.mc_runs, args.seed, args.initial)
    stress = _cost_stress(
        df,
        selected["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate,
        funding_cumulative=funding_cumulative,
    )
    gates = _candidate_gates(full_metrics, source_diff_pass, wfo, pbo, dsr, mc, stress, args)
    decision = _decision(gates)
    strict_ready = decision["ready_for_shadow"]

    report = {
        "schema_version": 1,
        "source_cuda_json": str(args.cuda_json),
        "rank": args.rank,
        "param_id": selected["performance"]["param_id"],
        "data": {
            "csv": str(csv_path),
            "period_start": period_start,
            "period_end": period_end,
            "rows": len(df),
        },
        "assumptions": {
            "market": "Binance USD-M perpetual",
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "funding_rate_per_8h": funding_rate,
            "funding_model": "actual_funding_events" if funding_csv is not None else obj.get("funding_model", "constant_per_8h"),
            "funding_rate_csv": str(funding_csv) if funding_csv is not None else None,
            "entry_delay_bars": entry_delay,
            "intrabar_policy_base": "conservative",
            "intrabar_policy_comparison": "optimistic",
            "drawdown_basis": obj.get("drawdown_basis"),
            "profit_factor_basis": obj.get("profit_factor_basis"),
        },
        "criteria": {
            "min_return": args.min_return,
            "min_pf": args.min_pf,
            "max_mdd": args.max_mdd,
            "min_trades": args.min_trades,
            "wfo_min_return": args.wfo_min_return,
            "wfo_min_pf": args.wfo_min_pf,
            "wfo_max_mdd": args.wfo_max_mdd,
            "wfo_min_trades": args.wfo_min_trades,
            "min_wfo_pass_ratio": args.min_wfo_pass_ratio,
            "max_pbo": args.max_pbo,
            "min_dsr": args.min_dsr,
            "min_mc_prob_positive": args.min_mc_prob_positive,
            "min_mc_return_p05": args.min_mc_return_p05,
            "min_stress_pass_ratio": args.min_stress_pass_ratio,
        },
        "candidate_universe": {
            "raw_candidates": len(raw_candidates),
            "unique_effective_candidates": len(candidates),
            "dsr_trials": dsr_trials,
            "dsr_trials_basis": dsr_trials_basis,
            "dedupe_basis": "canonical_parameters_ignore_inactive_or_unused_strategy_knobs",
        },
        "full_sample_cpu_reference": {"summary": _summary(full_metrics), "metrics": full_metrics},
        "source_period_cpu_reference": {
            "period_start": obj["period_start"],
            "period_end": obj["period_end"],
            "summary": _summary(source_metrics),
            "metrics": source_metrics,
        },
        "saved_cuda_performance": selected["performance"],
        "source_period_diff": source_diff,
        "source_period_diff_pass": source_diff_pass,
        "intrabar_policy_band": intrabar_policy_band,
        "fixed_candidate_wfo": wfo,
        "topk_cscv_pbo": pbo,
        "dsr": dsr,
        "monte_carlo": mc,
        "cost_stress": stress,
        "gates": gates,
        "decision": decision,
        "strict_ready_for_shadow": strict_ready,
    }
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        "summary "
        f"return={report['full_sample_cpu_reference']['summary']['return_pct']} "
        f"pf={report['full_sample_cpu_reference']['summary']['net_pf']} "
        f"mdd={report['full_sample_cpu_reference']['summary']['mdd_pct']} "
        f"wfo_pass_ratio={wfo['pass_ratio']} "
        f"pbo={pbo.get('pbo', 'n/a')} "
        f"dsr={dsr['dsr']} "
        f"stress_pass_ratio={stress['pass_ratio']} "
        f"strict_ready={strict_ready}"
    )
    return 2 if args.fail_on_hold and not strict_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
