from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_dsr_return_shape import _json_default, _num, _profit_factor, _return_distribution, _trade_distribution
from diff_cuda_cpu_reference import (
    _diff,
    _diff_pass,
    _intrabar_policy_comparison,
    _load_ohlcv,
    cpu_reference_backtest,
    load_funding_cumulative,
)
from validate_v2_strategy import _deflated_sharpe, _equity_returns, _max_drawdown_from_equity, _monte_carlo


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def _funding_event_times(path: Path) -> pd.Series:
    if not path.exists():
        raise SystemExit(f"Funding CSV not found: {path}")
    frame = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in frame.columns}
    time_col = cols.get("funding_time") or cols.get("timestamp") or cols.get("time")
    rate_col = cols.get("funding_rate") or cols.get("fundingrate")
    if time_col is None or rate_col is None:
        raise SystemExit(f"Funding CSV must contain funding_time/timestamp and funding_rate columns: {path}")
    numeric = pd.to_numeric(frame[time_col], errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(frame) * 0.9)):
        unit = "ms" if numeric.dropna().median() > 10_000_000_000 else "s"
        times = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        times = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    times = times.dropna().dt.tz_convert(None).sort_values()
    if times.empty:
        raise SystemExit(f"Funding CSV has no valid funding timestamps: {path}")
    return times


def _validate_funding_coverage(
    assumptions: dict[str, Any],
    df,
    *,
    max_lag_hours: float,
    allow_stale: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    funding_model = assumptions.get("funding_model")
    funding_csv = _resolve(assumptions.get("funding_rate_csv"), Path.cwd())
    if funding_model != "actual_funding_events":
        return {"enabled": False, "funding_model": funding_model, "status": "not_required"}, None
    if funding_csv is None:
        raise SystemExit("Manifest uses actual funding events but has no funding_rate_csv.")
    times = _funding_event_times(funding_csv)
    first_event = times.iloc[0]
    last_event = times.iloc[-1]
    visible_start = df.index.min()
    visible_end = df.index.max()
    lag_hours = (visible_end - last_event).total_seconds() / 3600.0
    starts_before_data = first_event <= visible_start
    lag_ok = lag_hours <= max_lag_hours
    coverage = {
        "enabled": True,
        "funding_model": funding_model,
        "funding_rate_csv": str(funding_csv),
        "first_funding_time": first_event.isoformat(),
        "last_funding_time": last_event.isoformat(),
        "visible_start": visible_start.isoformat(),
        "visible_end": visible_end.isoformat(),
        "lag_hours": round(lag_hours, 6),
        "max_lag_hours": max_lag_hours,
        "starts_before_visible_data": bool(starts_before_data),
        "status": "ok" if starts_before_data and lag_ok else "stale_or_incomplete",
        "allow_stale": bool(allow_stale),
    }
    if coverage["status"] != "ok" and not allow_stale:
        raise SystemExit(f"Funding coverage is stale or incomplete: {coverage}")
    return coverage, load_funding_cumulative(df, funding_csv)


def _combine_weighted_equity(equity_curves: list[np.ndarray], weights: np.ndarray, initial: float) -> np.ndarray:
    min_len = min((curve.size for curve in equity_curves), default=0)
    if min_len == 0:
        return np.array([initial], dtype=np.float64)
    combined = np.zeros(min_len, dtype=np.float64)
    for curve, weight in zip(equity_curves, weights):
        combined += curve[:min_len] * float(weight)
    return combined


def _gate(name: str, passed: bool, observed: Any, threshold: Any, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _execution_assumption_snapshot(assumptions: dict[str, Any], base: Path) -> dict[str, Any]:
    funding_csv = _resolve(assumptions.get("funding_rate_csv"), base)
    return {
        "market": assumptions.get("market"),
        "commission_rate": _num(assumptions.get("commission_rate"), 0.0),
        "slippage_rate": _num(assumptions.get("slippage_rate"), 0.0),
        "funding_rate_per_8h": _num(assumptions.get("funding_rate_per_8h"), 0.0),
        "funding_model": assumptions.get("funding_model"),
        "funding_rate_csv": str(funding_csv) if funding_csv is not None else None,
        "entry_delay_bars": int(assumptions.get("entry_delay_bars", 0)),
        "entry_price_policy": "next_bar_open",
        "intrabar_policy_base": assumptions.get("intrabar_policy_base", "conservative"),
        "drawdown_basis": "intrabar_mark_to_market_equity_curve",
        "profit_factor_reporting": "component_gross_and_net_pf_plus_portfolio_net_trade_pf",
        "cpu_reference_engine": "tools.diff_cuda_cpu_reference.cpu_reference_backtest",
        "htf_alignment": "4h_label_right_closed_left_ffill_with_runtime_no_lookahead_check",
    }


def _eval_component(
    df,
    component: dict[str, Any],
    assumptions: dict[str, Any],
    *,
    initial: float,
    funding_cumulative: np.ndarray | None,
    commission_mult: float = 1.0,
    slippage_mult: float = 1.0,
    funding_mult: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    scaled_funding = funding_cumulative * funding_mult if funding_cumulative is not None else None
    return cpu_reference_backtest(
        df,
        component["parameters"],
        initial_balance=initial,
        commission_rate=_num(assumptions.get("commission_rate"), 0.0005) * commission_mult,
        slippage_rate=_num(assumptions.get("slippage_rate"), 0.0002) * slippage_mult,
        entry_delay_bars=int(assumptions.get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h"), 0.0) * funding_mult,
        include_equity=True,
        include_trades=True,
        funding_cumulative=scaled_funding,
    )


def _component_source_cuda_diff(
    df,
    component: dict[str, Any],
    assumptions: dict[str, Any],
    *,
    funding_cumulative: np.ndarray | None,
    pnl_tol: float,
    pf_tol: float,
    mdd_tol: float,
    trade_tol: int,
) -> dict[str, Any]:
    cuda_performance = component.get("selected_train_performance", {})
    initial = _num(cuda_performance.get("initial_balance"), _num(assumptions.get("initial_balance"), 10_000.0))
    cpu = cpu_reference_backtest(
        df,
        component["parameters"],
        initial_balance=initial,
        commission_rate=_num(assumptions.get("commission_rate"), 0.0005),
        slippage_rate=_num(assumptions.get("slippage_rate"), 0.0002),
        entry_delay_bars=int(assumptions.get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h"), 0.0),
        funding_cumulative=funding_cumulative,
    )
    optimistic = cpu_reference_backtest(
        df,
        component["parameters"],
        initial_balance=initial,
        commission_rate=_num(assumptions.get("commission_rate"), 0.0005),
        slippage_rate=_num(assumptions.get("slippage_rate"), 0.0002),
        entry_delay_bars=int(assumptions.get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h"), 0.0),
        intrabar_policy="optimistic",
        funding_cumulative=funding_cumulative,
    )
    diff = _diff(cpu, cuda_performance)
    return {
        "component_id": component.get("component_id"),
        "source_profile": component.get("source_profile"),
        "source_cuda_results": component.get("source_cuda_results"),
        "rank": component.get("rank"),
        "param_id": component.get("param_id"),
        "cpu_reference": cpu,
        "cuda_performance": cuda_performance,
        "diff": diff,
        "diff_pass": _diff_pass(diff, pnl_tol, pf_tol, mdd_tol, trade_tol),
        "tolerances": {
            "pnl_pct": pnl_tol,
            "pf": pf_tol,
            "mdd_pct": mdd_tol,
            "trades": trade_tol,
        },
        "intrabar_policy_comparison": _intrabar_policy_comparison(cpu, optimistic),
    }


def _portfolio_eval(
    manifest: dict[str, Any],
    df,
    assumptions: dict[str, Any],
    *,
    funding_cumulative: np.ndarray | None,
    initial: float,
    commission_mult: float = 1.0,
    slippage_mult: float = 1.0,
    funding_mult: float = 1.0,
) -> dict[str, Any]:
    equity_curves: list[np.ndarray] = []
    all_trades: list[dict[str, Any]] = []
    component_rows = []
    weights = []
    for component in manifest.get("components", []):
        weight = _num(component.get("component_weight"))
        metrics, equity, trades = _eval_component(
            df,
            component,
            assumptions,
            initial=initial * weight,
            funding_cumulative=funding_cumulative,
            commission_mult=commission_mult,
            slippage_mult=slippage_mult,
            funding_mult=funding_mult,
        )
        equity_curves.append(equity)
        weights.append(weight)
        all_trades.extend(trades)
        component_rows.append(
            {
                "component_id": component.get("component_id"),
                "weight": weight,
                "source_profile": component.get("source_profile"),
                "rank": component.get("rank"),
                "param_id": component.get("param_id"),
                "metrics": metrics,
            }
        )
    combined = _combine_weighted_equity(equity_curves, np.ones(len(equity_curves), dtype=np.float64), initial)
    final_balance = float(combined[-1]) if combined.size else initial
    pnl = np.array([_num(row.get("net_pnl")) for row in all_trades], dtype=np.float64)
    summary = {
        "final_balance": round(final_balance, 6),
        "return_pct": round((final_balance - initial) / initial * 100.0, 6) if initial else 0.0,
        "mdd_pct": round(_max_drawdown_from_equity(combined), 6),
        "trades": len(all_trades),
        "trade_profit_factor": _profit_factor(pnl) if pnl.size else 0.0,
    }
    return {
        "summary": summary,
        "equity": combined,
        "trades": all_trades,
        "components": component_rows,
    }


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve(args.portfolio_manifest, Path.cwd())
    manifest = _load_json(manifest_path)
    assumptions = manifest.get("assumptions", {})
    execution_assumptions = _execution_assumption_snapshot(assumptions, Path.cwd())
    df = _load_ohlcv(_resolve(args.csv, Path.cwd()), args.start, args.end)
    funding_coverage, funding_cumulative = _validate_funding_coverage(
        assumptions,
        df,
        max_lag_hours=args.max_funding_lag_hours,
        allow_stale=args.allow_stale_funding,
    )
    initial = _num(assumptions.get("initial_balance"), 10_000.0)
    base = _portfolio_eval(
        manifest,
        df,
        assumptions,
        funding_cumulative=funding_cumulative,
        initial=initial,
    )
    dsr = _deflated_sharpe(base["equity"], args.n_trials)
    mc = _monte_carlo(base["equity"], args.mc_runs, args.seed, initial)
    returns = _equity_returns(base["equity"])
    source_cuda_diffs = [
        _component_source_cuda_diff(
            df,
            component,
            assumptions,
            funding_cumulative=funding_cumulative,
            pnl_tol=args.diff_pnl_tol,
            pf_tol=args.diff_pf_tol,
            mdd_tol=args.diff_mdd_tol,
            trade_tol=args.diff_trade_tol,
        )
        for component in manifest.get("components", [])
    ]
    source_cuda_diff_summary = {
        "component_count": len(source_cuda_diffs),
        "diff_pass_count": sum(1 for row in source_cuda_diffs if row.get("diff_pass") is True),
        "all_pass": bool(source_cuda_diffs) and all(row.get("diff_pass") is True for row in source_cuda_diffs),
        "failed_components": [row.get("component_id") for row in source_cuda_diffs if row.get("diff_pass") is not True],
        "tolerances": {
            "pnl_pct": args.diff_pnl_tol,
            "pf": args.diff_pf_tol,
            "mdd_pct": args.diff_mdd_tol,
            "trades": args.diff_trade_tol,
        },
    }
    stress_rows = []
    for commission_mult in args.stress_commission_mult:
        for slippage_mult in args.stress_slippage_mult:
            for funding_mult in args.stress_funding_mult:
                row = _portfolio_eval(
                    manifest,
                    df,
                    assumptions,
                    funding_cumulative=funding_cumulative,
                    initial=initial,
                    commission_mult=commission_mult,
                    slippage_mult=slippage_mult,
                    funding_mult=funding_mult,
                )
                summary = row["summary"]
                passed = (
                    summary["return_pct"] >= args.min_return
                    and _num(summary["trade_profit_factor"]) >= args.min_pf
                    and summary["mdd_pct"] <= args.max_mdd
                    and summary["trades"] >= args.min_trades
                )
                stress_rows.append(
                    {
                        "commission_mult": commission_mult,
                        "slippage_mult": slippage_mult,
                        "funding_mult": funding_mult,
                        "summary": summary,
                        "pass": passed,
                    }
                )
    stress = {
        "scenarios": stress_rows,
        "pass_ratio": round(sum(1 for row in stress_rows if row["pass"]) / len(stress_rows), 6) if stress_rows else 0.0,
    }
    gates = [
        _gate("return_pct", base["summary"]["return_pct"] >= args.min_return, base["summary"]["return_pct"], f">= {args.min_return}", "Portfolio current-component return."),
        _gate("trade_profit_factor", _num(base["summary"]["trade_profit_factor"]) >= args.min_pf, base["summary"]["trade_profit_factor"], f">= {args.min_pf}", "Portfolio current-component trade PF."),
        _gate("mdd_pct", base["summary"]["mdd_pct"] <= args.max_mdd, base["summary"]["mdd_pct"], f"<= {args.max_mdd}", "Portfolio current-component MDD."),
        _gate("trades", base["summary"]["trades"] >= args.min_trades, base["summary"]["trades"], f">= {args.min_trades}", "Portfolio current-component trade count."),
        _gate("dsr", _num(dsr.get("dsr")) >= args.min_dsr, dsr.get("dsr"), f">= {args.min_dsr}", "Current-component DSR."),
        _gate("mc_return_p05", _num(mc.get("return_pct_p05", -1e9)) >= args.min_mc_return_p05, mc.get("return_pct_p05"), f">= {args.min_mc_return_p05}", "Current-component MC p05."),
        _gate("stress_pass_ratio", stress["pass_ratio"] >= args.min_stress_pass_ratio, stress["pass_ratio"], f">= {args.min_stress_pass_ratio}", "Current-component cost stress."),
        _gate(
            "source_cpu_cuda_diff",
            source_cuda_diff_summary["all_pass"],
            source_cuda_diff_summary,
            "all active components diff-pass CPU reference vs CUDA source rows",
            "Active component CUDA/fast-search rows must match the CPU reference engine before shadow observation.",
        ),
    ]
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    return {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "period": {"start": args.start, "end": args.end, "rows": len(df)},
        "funding_coverage": funding_coverage,
        "criteria": {
            "min_return": args.min_return,
            "min_pf": args.min_pf,
            "max_mdd": args.max_mdd,
            "min_trades": args.min_trades,
            "min_dsr": args.min_dsr,
            "min_mc_return_p05": args.min_mc_return_p05,
            "min_stress_pass_ratio": args.min_stress_pass_ratio,
        },
        "summary": base["summary"],
        "execution_assumptions": execution_assumptions,
        "source_cpu_cuda_diff": source_cuda_diff_summary,
        "dsr": dsr,
        "monte_carlo": mc,
        "cost_stress": stress,
        "return_distribution": _return_distribution(returns),
        "trade_distribution": _trade_distribution(base["trades"]),
        "gates": gates,
        "decision": {
            "status": "CURRENT_COMPONENTS_VALIDATED_FOR_SHADOW_OBSERVATION" if not failed else "HOLD_AUTOMATED_PAPER",
            "ready_for_shadow": not failed,
            "ready_for_paper": False,
            "failed_gates": failed,
            "rationale": (
                "Current component set passed diagnostic gates for no-order shadow observation only."
                if not failed
                else "Current component set remains held until failed gates pass."
            ),
        },
        "components": base["components"],
        "component_source_cpu_cuda_diffs": source_cuda_diffs,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Shadow Manifest Validation",
        "",
        f"- Source manifest: `{result['source_manifest']}`",
        f"- Decision: `{result['decision']['status']}`",
        f"- Ready for shadow: `{str(result['decision']['ready_for_shadow']).lower()}`",
        f"- Ready for paper: `false`",
        f"- Funding coverage: `{result.get('funding_coverage', {}).get('status')}`",
        f"- Source CPU/CUDA diff: `{result.get('source_cpu_cuda_diff', {}).get('diff_pass_count')}/{result.get('source_cpu_cuda_diff', {}).get('component_count')}` pass",
        "",
        "| return | mdd | trades | pf | dsr | mc_p05 | stress | failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {result['summary']['return_pct']} | {result['summary']['mdd_pct']} | {result['summary']['trades']} | "
            f"{result['summary']['trade_profit_factor']} | {result['dsr'].get('dsr')} | "
            f"{result['monte_carlo'].get('return_pct_p05')} | {result['cost_stress'].get('pass_ratio')} | "
            f"{','.join(result['decision']['failed_gates'])} |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate current component params in a portfolio shadow manifest.")
    parser.add_argument("--portfolio-manifest", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument("--start", default="2019-12-24")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--n-trials", type=int, default=1771)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--min-mc-return-p05", type=float, default=0.0)
    parser.add_argument("--min-stress-pass-ratio", type=float, default=0.5)
    parser.add_argument("--max-funding-lag-hours", type=float, default=12.0)
    parser.add_argument("--allow-stale-funding", action="store_true")
    parser.add_argument("--stress-commission-mult", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--stress-slippage-mult", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--stress-funding-mult", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--diff-pnl-tol", type=float, default=0.01)
    parser.add_argument("--diff-pf-tol", type=float, default=0.01)
    parser.add_argument("--diff-mdd-tol", type=float, default=0.01)
    parser.add_argument("--diff-trade-tol", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    result = validate_manifest(args)
    if args.out_json:
        out_json = _resolve(args.out_json, Path.cwd())
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    if args.out_md:
        out_md = _resolve(args.out_md, Path.cwd())
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, result)
    print(
        json.dumps(
            {
                "decision": result["decision"]["status"],
                "ready_for_shadow": result["decision"]["ready_for_shadow"],
                "ready_for_paper": result["decision"]["ready_for_paper"],
                "return_pct": result["summary"]["return_pct"],
                "dsr": result["dsr"].get("dsr"),
                "stress_pass_ratio": result["cost_stress"].get("pass_ratio"),
                "source_cpu_cuda_diff": result.get("source_cpu_cuda_diff"),
                "failed_gates": result["decision"]["failed_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
