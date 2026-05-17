from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from diff_cuda_cpu_reference import _diff, _diff_pass, _load_ohlcv, _num, load_funding_cumulative
from validate_cuda_candidate import (
    _candidate_gates,
    _cost_stress,
    _default_wfo_min_trades,
    _dedupe_candidates,
    _decision,
    _fixed_candidate_wfo,
    _effective_dsr_trials,
    _intrabar_policy_band,
    _json_default,
    _load_results,
    _run,
    _summary,
    _topk_cscv_pbo,
)
from validate_v2_strategy import _deflated_sharpe, _monte_carlo


def _resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _parse_ranks(value: str | None) -> set[int] | None:
    if not value:
        return None
    ranks: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"Invalid rank range: {part}")
            ranks.update(range(start, end + 1))
        else:
            ranks.add(int(part))
    return ranks


def _source_period_diff_pass(
    source_df: pd.DataFrame,
    row: dict[str, Any],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    funding_cumulative: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    source_metrics = _run(
        source_df,
        row["parameters"],
        args.initial,
        commission_rate,
        slippage_rate,
        args.entry_delay_bars,
        funding_rate_per_8h,
        funding_cumulative=funding_cumulative,
    )
    diff = _diff(source_metrics, row["performance"])
    passed = _diff_pass(diff, args.pnl_tol, args.pf_tol, args.mdd_tol, args.trade_tol)
    return source_metrics, diff, passed


def _validate_row(
    obj: dict[str, Any],
    row: dict[str, Any],
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    pbo: dict[str, Any],
    args: argparse.Namespace,
    commission_rate: float,
    slippage_rate: float,
    funding_rate_per_8h: float,
    funding_csv: Path | None = None,
    funding_cumulative: np.ndarray | None = None,
    source_funding_cumulative: np.ndarray | None = None,
) -> dict[str, Any]:
    full_metrics, equity = _run(
        df,
        row["parameters"],
        args.initial,
        commission_rate,
        slippage_rate,
        args.entry_delay_bars,
        funding_rate_per_8h,
        include_equity=True,
        funding_cumulative=funding_cumulative,
    )
    source_metrics, source_diff, source_diff_pass = _source_period_diff_pass(
        source_df,
        row,
        args,
        commission_rate,
        slippage_rate,
        funding_rate_per_8h,
        source_funding_cumulative,
    )
    intrabar_policy_band = _intrabar_policy_band(
        df,
        row["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate_per_8h,
        full_metrics,
        funding_cumulative=funding_cumulative,
    )
    wfo = _fixed_candidate_wfo(
        df,
        row["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate_per_8h,
        funding_csv=funding_csv,
    )
    dsr = _deflated_sharpe(equity, max(2, args.num_dsr_trials))
    mc = _monte_carlo(equity, args.mc_runs, args.seed + int(row["rank"]), args.initial)
    stress = _cost_stress(
        df,
        row["parameters"],
        args,
        commission_rate,
        slippage_rate,
        funding_rate_per_8h,
        funding_cumulative=funding_cumulative,
    )
    gates = _candidate_gates(full_metrics, source_diff_pass, wfo, pbo, dsr, mc, stress, args)
    decision = _decision(gates)
    return {
        "rank": int(row["rank"]),
        "param_id": row["performance"]["param_id"],
        "saved_cuda_summary": _summary(row["performance"]),
        "saved_cuda_strict_pass": bool(row["performance"].get("strict_pass", False)),
        "full_sample_cpu_reference": {"summary": _summary(full_metrics), "metrics": full_metrics},
        "source_period_cpu_reference": {
            "period_start": obj["period_start"],
            "period_end": obj["period_end"],
            "summary": _summary(source_metrics),
            "metrics": source_metrics,
        },
        "source_period_diff": source_diff,
        "source_period_diff_pass": source_diff_pass,
        "intrabar_policy_band": intrabar_policy_band,
        "fixed_candidate_wfo": wfo,
        "dsr": dsr,
        "monte_carlo": mc,
        "cost_stress": stress,
        "gates": gates,
        "decision": decision,
        "strict_ready_for_shadow": decision["ready_for_shadow"],
    }


def _candidate_score(candidate: dict[str, Any]) -> tuple[int, float, float, float, int]:
    summary = candidate["full_sample_cpu_reference"]["summary"]
    failed = len(candidate["decision"]["failed_gates"])
    return (
        -failed,
        _num(summary["return_pct"]),
        _num(summary["net_pf"]),
        -_num(summary["mdd_pct"]),
        int(summary["trades"]),
    )


def _write_individual_report(
    out_dir: Path,
    base_payload: dict[str, Any],
    candidate: dict[str, Any],
) -> Path:
    path = out_dir / f"cuda_candidate_validation_rank{candidate['rank']}.json"
    report = dict(base_payload)
    report.update(
        {
            "rank": candidate["rank"],
            "param_id": candidate["param_id"],
            "saved_cuda_summary": candidate["saved_cuda_summary"],
            "saved_cuda_strict_pass": candidate["saved_cuda_strict_pass"],
            "full_sample_cpu_reference": candidate["full_sample_cpu_reference"],
            "source_period_cpu_reference": candidate["source_period_cpu_reference"],
            "source_period_diff": candidate["source_period_diff"],
            "source_period_diff_pass": candidate["source_period_diff_pass"],
            "intrabar_policy_band": candidate["intrabar_policy_band"],
            "fixed_candidate_wfo": candidate["fixed_candidate_wfo"],
            "topk_cscv_pbo": base_payload["topk_cscv_pbo"],
            "dsr": candidate["dsr"],
            "monte_carlo": candidate["monte_carlo"],
            "cost_stress": candidate["cost_stress"],
            "gates": candidate["gates"],
            "decision": candidate["decision"],
            "strict_ready_for_shadow": candidate["strict_ready_for_shadow"],
        }
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-validate saved CUDA candidates and emit a promotion manifest."
    )
    parser.add_argument("--cuda-json", required=True, type=Path)
    parser.add_argument("--csv", default=Path("data/BTCUSDT_1h.csv"), type=Path)
    parser.add_argument("--start", default=None, help="Override validation start; defaults to CUDA JSON period_start.")
    parser.add_argument("--end", default=None, help="Override validation end; defaults to CUDA JSON period_end.")
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks/ranges, e.g. 1,3,5-8.")
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--strict-only", action="store_true", help="Only validate rows that passed saved CUDA strict gates.")
    parser.add_argument("--keep-duplicate-selected", action="store_true", help="Validate duplicate effective strategies instead of keeping only the first occurrence.")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--dsr-trials", type=int, default=None, help="Override DSR multiple-testing trial count; defaults to total_param_combinations when available.")
    parser.add_argument("--funding-csv", type=Path, default=None, help="Optional Binance funding-rate event CSV for actual funding charges.")
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--entry-delay-bars", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-fold-rows", type=int, default=1000)
    parser.add_argument("--cscv-splits", type=int, default=6)
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
    parser.add_argument("--pnl-tol", type=float, default=0.01)
    parser.add_argument("--pf-tol", type=float, default=0.01)
    parser.add_argument("--mdd-tol", type=float, default=0.01)
    parser.add_argument("--trade-tol", type=int, default=0)
    parser.add_argument("--out", default=Path("wfa_optimized_params_output/cuda_candidate_batch_manifest.json"), type=Path)
    parser.add_argument("--write-individual-reports", action="store_true")
    parser.add_argument("--fail-on-no-promote", action="store_true")
    args = parser.parse_args()
    if args.wfo_max_mdd is None:
        args.wfo_max_mdd = args.max_mdd

    obj = _load_results(_resolve_path(args.cuda_json))
    csv_path = _resolve_path(args.csv)
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
    funding_csv = _resolve_path(funding_csv) if funding_csv is not None else None
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    source_funding_cumulative = load_funding_cumulative(source_df, funding_csv) if funding_csv is not None else None
    entry_delay = int(obj.get("entry_delay_bars", 1)) if args.entry_delay_bars is None else args.entry_delay_bars
    args.entry_delay_bars = entry_delay

    rank_filter = _parse_ranks(args.ranks)
    rows = [row for row in obj.get("results", []) if int(row["rank"]) <= args.max_rank]
    if rank_filter is not None:
        rows = [row for row in rows if int(row["rank"]) in rank_filter]
    if args.strict_only:
        rows = [row for row in rows if bool(row.get("performance", {}).get("strict_pass", False))]
    raw_selected_count = len(rows)
    if not args.keep_duplicate_selected:
        rows = _dedupe_candidates(rows)
    if not rows:
        raise SystemExit("No candidates selected for validation.")

    raw_candidates_for_pbo = obj.get("results", [])[: args.max_candidates]
    candidates_for_pbo = _dedupe_candidates(raw_candidates_for_pbo)
    args.num_dsr_trials, dsr_trials_basis = _effective_dsr_trials(obj, candidates_for_pbo, args.dsr_trials)
    pbo = _topk_cscv_pbo(
        df,
        candidates_for_pbo,
        args,
        commission_rate,
        slippage_rate,
        funding_rate,
        funding_csv=funding_csv,
    )

    base_payload = {
        "schema_version": 1,
        "source_cuda_json": str(_resolve_path(args.cuda_json)),
        "data": {"csv": str(csv_path), "period_start": period_start, "period_end": period_end, "rows": len(df)},
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
        "selection": {
            "max_rank": args.max_rank,
            "ranks": sorted(rank_filter) if rank_filter else None,
            "strict_only": args.strict_only,
            "raw_selected_count": raw_selected_count,
            "deduped_selected": not args.keep_duplicate_selected,
            "selected_ranks": [int(row["rank"]) for row in rows],
        },
        "candidate_universe": {
            "raw_candidates": len(raw_candidates_for_pbo),
            "unique_effective_candidates": len(candidates_for_pbo),
            "dsr_trials": args.num_dsr_trials,
            "dsr_trials_basis": dsr_trials_basis,
            "dedupe_basis": "canonical_parameters_ignore_inactive_or_unused_strategy_knobs",
        },
        "topk_cscv_pbo": pbo,
    }

    out_path = _resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    individual_dir = out_path.with_suffix("")
    if args.write_individual_reports:
        individual_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] validating rank {row['rank']} {row['performance']['param_id']}")
        try:
            candidate = _validate_row(
                obj,
                row,
                df,
                source_df,
                pbo,
                args,
                commission_rate,
                slippage_rate,
                funding_rate,
                funding_csv=funding_csv,
                funding_cumulative=funding_cumulative,
                source_funding_cumulative=source_funding_cumulative,
            )
            if args.write_individual_reports:
                report_path = _write_individual_report(individual_dir, base_payload, candidate)
                candidate["individual_report"] = str(report_path)
            candidates.append(candidate)
        except Exception as exc:  # noqa: BLE001 - batch mode should keep validating later ranks.
            candidates.append(
                {
                    "rank": int(row["rank"]),
                    "param_id": row["performance"].get("param_id"),
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "strict_ready_for_shadow": False,
                    "decision": {
                        "status": "HOLD_AUTOMATED_PAPER",
                        "ready_for_shadow": False,
                        "ready_for_paper": False,
                        "failed_gates": ["batch_validation_error"],
                    },
                }
            )

    promoted = [row for row in candidates if row.get("strict_ready_for_shadow")]
    failure_counts: Counter[str] = Counter()
    for row in candidates:
        failure_counts.update(row.get("decision", {}).get("failed_gates", []))

    ranked_candidates = sorted(
        [row for row in candidates if "full_sample_cpu_reference" in row],
        key=_candidate_score,
        reverse=True,
    )
    manifest = dict(base_payload)
    manifest.update(
        {
            "summary": {
                "validated": len(candidates),
                "promoted_to_shadow": len(promoted),
                "held": len(candidates) - len(promoted),
                "failure_counts": dict(failure_counts),
                "best_rank_by_gate_count_then_return": ranked_candidates[0]["rank"] if ranked_candidates else None,
                "decision": "PROMOTE_TO_SHADOW" if promoted else "HOLD_AUTOMATED_PAPER",
                "ready_for_paper": False,
            },
            "candidates": candidates,
        }
    )
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        "batch summary "
        f"validated={manifest['summary']['validated']} "
        f"promoted_to_shadow={manifest['summary']['promoted_to_shadow']} "
        f"decision={manifest['summary']['decision']} "
        f"top_failures={dict(failure_counts.most_common(5))}"
    )
    return 2 if args.fail_on_no_promote and not promoted else 0


if __name__ == "__main__":
    raise SystemExit(main())
