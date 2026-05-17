from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative
from tools.validate_v2_strategy import _deflated_sharpe, _monte_carlo, _max_drawdown_from_equity


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj == float("inf"):
        return "inf"
    return str(obj)


def _num(value: Any, default: float = 0.0) -> float:
    if value == "inf":
        return float("inf")
    if value is None:
        return default
    return float(value)


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _latest_result(output_dir: Path, profile: str, started: float, known: set[Path]) -> Path | None:
    candidates = []
    new_paths = []
    for path in output_dir.glob(f"top_results_BTCUSDT_{profile}_*.json"):
        if path in known:
            continue
        new_paths.append(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= started - 1.0:
            candidates.append((mtime, path))
    if not candidates:
        fallback = []
        for path in new_paths:
            try:
                fallback.append((path.stat().st_mtime, path))
            except OSError:
                continue
        return sorted(fallback, reverse=True)[0][1] if fallback else None
    return sorted(candidates, reverse=True)[0][1]


def _folds(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_bars: int,
    embargo_bars: int,
    mode: str,
) -> list[dict[str, Any]]:
    first = pd.to_datetime(start, utc=True).tz_convert(None).floor("h")
    last = pd.to_datetime(end, utc=True).tz_convert(None)
    if len(str(end).strip()) == 10 and str(end).count("-") == 2:
        last = last + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    last = last.floor("h")
    rows = []
    if mode not in {"rolling", "anchored"}:
        raise ValueError(f"Unsupported WFO fold mode: {mode}")
    fold_start = first
    fold_index = 0
    while True:
        if mode == "anchored":
            train_start = first
            train_end_exclusive = first + pd.DateOffset(months=train_months + fold_index * step_months)
        else:
            train_start = fold_start
            train_end_exclusive = fold_start + pd.DateOffset(months=train_months)
        test_end_exclusive = train_end_exclusive + pd.DateOffset(months=test_months)
        if test_end_exclusive - pd.Timedelta(hours=1) > last:
            break
        train_end = train_end_exclusive - pd.Timedelta(hours=purge_bars + 1)
        test_start = train_end_exclusive + pd.Timedelta(hours=embargo_bars)
        test_end = test_end_exclusive - pd.Timedelta(hours=1)
        if train_end > train_start and test_end > test_start:
            rows.append(
                {
                    "fold": len(rows) + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
        fold_start = fold_start + pd.DateOffset(months=step_months)
        fold_index += 1
    return rows


def _run_cuda_train_search(args: argparse.Namespace, fold: dict[str, Any]) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    known = set(output_dir.glob(f"top_results_BTCUSDT_{args.profile}_*.json"))
    runner = ROOT / "tools" / "run_cuda_strategy_search.py"
    cmd = [
        args.python,
        str(runner),
        "--profile",
        args.profile,
        "--csv",
        str(args.csv),
        "--start",
        _fmt(fold["train_start"]),
        "--end",
        _fmt(fold["train_end"]),
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--commission",
        str(args.commission),
        "--slippage",
        str(args.slippage),
        "--entry-delay-bars",
        str(args.entry_delay_bars),
        "--funding-rate-per-8h",
        str(args.funding_rate_per_8h),
        "--strict-min-return",
        str(args.train_min_return),
        "--strict-min-pf",
        str(args.train_min_pf),
        "--strict-max-mdd",
        str(args.train_max_mdd),
        "--strict-min-trades",
        str(args.train_min_trades),
        "--rank-metric",
        args.rank_metric,
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--python",
        args.python,
    ]
    if args.funding_csv is not None:
        cmd.extend(["--funding-csv", str(args.funding_csv)])
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    result_path = _latest_result(output_dir, args.profile, started, known)
    if proc.returncode != 0:
        raise RuntimeError(
            "CUDA train search failed for fold "
            f"{fold['fold']} with exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    if result_path is None:
        return {
            "results": [],
            "total_param_combinations": None,
            "_result_path": None,
            "_missing_result_json": True,
            "_missing_result_stdout_tail": proc.stdout[-4000:],
            "_missing_result_stderr_tail": proc.stderr[-4000:],
        }
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    obj["_result_path"] = str(result_path)
    return obj


def _select_candidates(search_obj: dict[str, Any], strict_only: bool, limit: int) -> list[dict[str, Any]]:
    selected = []
    for row in search_obj.get("results", []):
        if strict_only and not bool(row.get("performance", {}).get("strict_pass", False)):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _select_candidate(search_obj: dict[str, Any], strict_only: bool) -> dict[str, Any] | None:
    selected = _select_candidates(search_obj, strict_only, 1)
    return selected[0] if selected else None


def _strict_reasons(performance: dict[str, Any]) -> list[str]:
    reasons = performance.get("strict_reasons") or []
    if isinstance(reasons, list):
        return [str(reason) for reason in reasons]
    return [str(reasons)]


def _compact_train_performance(performance: dict[str, Any] | None) -> dict[str, Any] | None:
    if not performance:
        return None
    return {
        "return_pct": round(_num(performance.get("total_net_pnl_percentage")), 4),
        "net_pf": round(_num(performance.get("net_profit_factor", performance.get("profit_factor"))), 4),
        "gross_pf": round(_num(performance.get("gross_profit_factor")), 4),
        "mdd_pct": round(_num(performance.get("max_drawdown_percentage")), 4),
        "trades": int(performance.get("num_trades", 0)),
        "win_rate_pct": round(_num(performance.get("win_rate_percentage")), 4),
        "strict_pass": bool(performance.get("strict_pass", False)),
        "strict_reasons": _strict_reasons(performance),
        "rank_metric": performance.get("rank_metric"),
        "rank_score": round(_num(performance.get("rank_score")), 6),
        "error": bool(performance.get("error", False)),
    }


def _train_search_diagnostics(search_obj: dict[str, Any]) -> dict[str, Any]:
    results = list(search_obj.get("results", []))
    top = results[0] if results else None
    top_performance = top.get("performance", {}) if top else {}
    failure_counts: dict[str, int] = {}
    strict_pass_count = 0
    for item in results:
        performance = item.get("performance", {}) or {}
        if bool(performance.get("strict_pass", False)):
            strict_pass_count += 1
            continue
        reasons = _strict_reasons(performance) or ["unknown"]
        for reason in reasons:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "train_result_json": search_obj.get("_result_path"),
        "train_result_missing": bool(search_obj.get("_missing_result_json", False)),
        "train_missing_result_reason": "no_cuda_result_json" if search_obj.get("_missing_result_json") else None,
        "train_total_param_combinations": search_obj.get("total_param_combinations"),
        "train_results_count": len(results),
        "train_strict_pass_count_top_k": strict_pass_count,
        "train_strict_failure_counts_top_k": dict(sorted(failure_counts.items())),
        "train_top_rank": int(top.get("rank")) if top else None,
        "train_top_param_id": (top.get("param_id") or top_performance.get("param_id")) if top else None,
        "train_top_performance": _compact_train_performance(top_performance),
    }


def _cpu_eval(
    csv: Path,
    fold: dict[str, Any],
    params: dict[str, Any],
    args: argparse.Namespace,
    *,
    commission: float,
    slippage: float,
    funding: float,
    funding_csv: Path | None = None,
    funding_multiplier: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    df = _load_ohlcv(csv, _fmt(fold["test_start"]), _fmt(fold["test_end"]))
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    if funding_cumulative is not None:
        funding_cumulative = funding_cumulative * funding_multiplier
    summary, equity, trades = cpu_reference_backtest(
        df,
        params,
        initial_balance=args.initial,
        commission_rate=commission,
        slippage_rate=slippage,
        entry_delay_bars=args.entry_delay_bars,
        funding_rate_per_8h=funding * funding_multiplier,
        include_equity=True,
        include_trades=True,
        funding_cumulative=funding_cumulative,
    )
    return summary, equity, trades


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": round(_num(metrics.get("total_net_pnl_percentage")), 4),
        "net_pf": round(_num(metrics.get("net_profit_factor", metrics.get("profit_factor"))), 4),
        "gross_pf": round(_num(metrics.get("gross_profit_factor")), 4),
        "mdd_pct": round(_num(metrics.get("max_drawdown_percentage")), 4),
        "trades": int(metrics.get("num_trades", 0)),
        "win_rate_pct": round(_num(metrics.get("win_rate_percentage")), 4),
    }


def _summary_from_equity_and_trades(equity: np.ndarray, trades: list[dict[str, Any]], initial: float) -> dict[str, Any]:
    final_balance = float(equity[-1]) if equity.size else initial
    net_profit = sum(max(0.0, float(trade.get("net_pnl", 0.0))) for trade in trades)
    net_loss = sum(abs(min(0.0, float(trade.get("net_pnl", 0.0)))) for trade in trades)
    gross_profit = sum(max(0.0, float(trade.get("gross_pnl", 0.0))) for trade in trades)
    gross_loss = sum(abs(min(0.0, float(trade.get("gross_pnl", 0.0)))) for trade in trades)
    wins = sum(1 for trade in trades if float(trade.get("net_pnl", 0.0)) > 0.0)
    return {
        "return_pct": round((final_balance - initial) / initial * 100.0, 4),
        "net_pf": round(net_profit / net_loss, 4) if net_loss > 0 else ("inf" if net_profit > 0 else 0.0),
        "gross_pf": round(gross_profit / gross_loss, 4) if gross_loss > 0 else ("inf" if gross_profit > 0 else 0.0),
        "mdd_pct": round(_max_drawdown_from_equity(equity), 4),
        "trades": len(trades),
        "win_rate_pct": round(wins / len(trades) * 100.0, 4) if trades else 0.0,
    }


def _scale_trade(trade: dict[str, Any], weight: float, component_index: int) -> dict[str, Any]:
    scaled = dict(trade)
    for key in (
        "gross_pnl",
        "net_pnl",
        "commission",
        "entry_commission",
        "exit_commission",
        "slippage_cost",
        "funding_cost",
    ):
        if key in scaled:
            try:
                scaled[key] = float(scaled[key]) * weight
            except (TypeError, ValueError):
                pass
    scaled["ensemble_weight"] = weight
    scaled["ensemble_component_index"] = component_index
    return scaled


def _selection_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    ensemble = row.get("selected_ensemble")
    if ensemble:
        return list(ensemble)
    if row.get("selected_parameters") is None:
        return []
    return [
        {
            "rank": row.get("selected_rank"),
            "param_id": row.get("selected_param_id"),
            "parameters": row.get("selected_parameters"),
            "performance": row.get("selected_train_performance"),
        }
    ]


def _eval_selection(
    csv: Path,
    fold: dict[str, Any],
    selections: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    commission: float,
    slippage: float,
    funding: float,
    funding_csv: Path | None = None,
    funding_multiplier: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if not selections:
        empty_summary = {"return_pct": 0.0, "net_pf": 0.0, "gross_pf": 0.0, "mdd_pct": 0.0, "trades": 0, "win_rate_pct": 0.0}
        return empty_summary, np.array([args.initial], dtype=np.float64), [], []

    if len(selections) == 1:
        summary, equity, trades = _cpu_eval(
            csv,
            fold,
            selections[0]["parameters"],
            args,
            commission=commission,
            slippage=slippage,
            funding=funding,
            funding_csv=funding_csv,
            funding_multiplier=funding_multiplier,
        )
        return _summary(summary), equity, trades, [{"weight": 1.0, "summary": _summary(summary)}]

    weights = np.full(len(selections), 1.0 / len(selections), dtype=np.float64)
    component_equities = []
    scaled_trades: list[dict[str, Any]] = []
    component_summaries = []
    for idx, (selection, weight) in enumerate(zip(selections, weights)):
        summary, equity, trades = _cpu_eval(
            csv,
            fold,
            selection["parameters"],
            args,
            commission=commission,
            slippage=slippage,
            funding=funding,
            funding_csv=funding_csv,
            funding_multiplier=funding_multiplier,
        )
        component_equities.append(equity)
        component_summary = _summary(summary)
        component_summaries.append(
            {
                "rank": selection.get("rank"),
                "param_id": selection.get("param_id") or selection.get("performance", {}).get("param_id"),
                "weight": round(float(weight), 6),
                "summary": component_summary,
            }
        )
        for trade in trades:
            scaled_trades.append(_scale_trade(trade, float(weight), idx))

    min_len = min((equity.size for equity in component_equities), default=0)
    if min_len == 0:
        combined = np.array([args.initial], dtype=np.float64)
    else:
        combined = np.zeros(min_len, dtype=np.float64)
        for equity, weight in zip(component_equities, weights):
            combined += equity[:min_len] * float(weight)
    return _summary_from_equity_and_trades(combined, scaled_trades, args.initial), combined, scaled_trades, component_summaries


def _combine_equity(equity_curves: list[np.ndarray], initial: float) -> np.ndarray:
    combined = [float(initial)]
    current = float(initial)
    for curve in equity_curves:
        if curve.size < 2:
            continue
        returns = np.divide(
            curve[1:] - curve[:-1],
            curve[:-1],
            out=np.zeros(curve.size - 1, dtype=np.float64),
            where=curve[:-1] != 0,
        )
        for ret in returns:
            current *= 1.0 + float(ret)
            combined.append(current)
    return np.array(combined, dtype=np.float64)


def _aggregate(folds: list[dict[str, Any]], equity_curves: list[np.ndarray], trades_by_fold: list[list[dict[str, Any]]], initial: float) -> dict[str, Any]:
    combined_equity = _combine_equity(equity_curves, initial)
    final_balance = float(combined_equity[-1]) if combined_equity.size else initial
    all_trades = [trade for trades in trades_by_fold for trade in trades]
    fold_summaries = [row.get("test_summary") or row.get("summary") for row in folds]
    fold_summaries = [row for row in fold_summaries if row]
    net_profit = sum(max(0.0, float(trade.get("net_pnl", 0.0))) for trade in all_trades)
    net_loss = sum(abs(min(0.0, float(trade.get("net_pnl", 0.0)))) for trade in all_trades)
    gross_profit = sum(max(0.0, float(trade.get("gross_pnl", 0.0))) for trade in all_trades)
    gross_loss = sum(abs(min(0.0, float(trade.get("gross_pnl", 0.0)))) for trade in all_trades)
    wins = sum(1 for trade in all_trades if float(trade.get("net_pnl", 0.0)) > 0.0)
    return {
        "final_balance": round(final_balance, 4),
        "return_pct": round((final_balance - initial) / initial * 100.0, 4),
        "net_pf": round(net_profit / net_loss, 4) if net_loss > 0 else ("inf" if net_profit > 0 else 0.0),
        "gross_pf": round(gross_profit / gross_loss, 4) if gross_loss > 0 else ("inf" if gross_profit > 0 else 0.0),
        "mdd_pct": round(_max_drawdown_from_equity(combined_equity), 4),
        "trades": len(all_trades),
        "win_rate_pct": round(wins / len(all_trades) * 100.0, 4) if all_trades else 0.0,
        "fold_return_median_pct": round(float(np.median([row["return_pct"] for row in fold_summaries])), 4) if fold_summaries else 0.0,
        "fold_pf_median": round(float(np.median([_num(row["net_pf"]) for row in fold_summaries])), 4) if fold_summaries else 0.0,
    }


def _passes_test(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        _num(summary["return_pct"]) >= args.test_min_return
        and _num(summary["net_pf"]) >= args.test_min_pf
        and _num(summary["mdd_pct"]) <= args.test_max_mdd
        and int(summary["trades"]) >= args.test_min_trades
    )


def _gate(name: str, passed: bool, observed: Any, threshold: Any, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _stress_selector(
    fold_rows: list[dict[str, Any]],
    csv: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = []
    for commission_mult in args.stress_commission_mult:
        for slippage_mult in args.stress_slippage_mult:
            for funding_mult in args.stress_funding_mult:
                equity_curves = []
                trades_by_fold = []
                test_rows = []
                for row in fold_rows:
                    selections = _selection_items(row)
                    if not selections:
                        continue
                    fold = {
                        "fold": row["fold"],
                        "test_start": pd.to_datetime(row["test_start"]),
                        "test_end": pd.to_datetime(row["test_end"]),
                    }
                    test_summary, equity, trades, _components = _eval_selection(
                        csv,
                        fold,
                        selections,
                        args,
                        commission=args.commission * commission_mult,
                        slippage=args.slippage * slippage_mult,
                        funding=args.funding_rate_per_8h,
                        funding_csv=args.funding_csv,
                        funding_multiplier=funding_mult,
                    )
                    test_rows.append({"summary": test_summary, "pass": _passes_test(test_summary, args)})
                    equity_curves.append(equity)
                    trades_by_fold.append(trades)
                aggregate = _aggregate(test_rows, equity_curves, trades_by_fold, args.initial)
                passed = (
                    aggregate["return_pct"] >= args.min_return
                    and _num(aggregate["net_pf"]) >= args.min_pf
                    and aggregate["mdd_pct"] <= args.max_mdd
                    and aggregate["trades"] >= args.min_trades
                    and (sum(1 for row in test_rows if row["pass"]) / len(test_rows) if test_rows else 0.0) >= args.min_test_pass_ratio
                )
                rows.append(
                    {
                        "commission_mult": commission_mult,
                        "slippage_mult": slippage_mult,
                        "funding_mult": funding_mult,
                        "aggregate": aggregate,
                        "test_pass_ratio": round(sum(1 for row in test_rows if row["pass"]) / len(test_rows), 4) if test_rows else 0.0,
                        "pass": passed,
                    }
                )
    return {
        "scenarios": rows,
        "pass_ratio": round(sum(1 for row in rows if row["pass"]) / len(rows), 4) if rows else 0.0,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    is_ensemble = result.get("assumptions", {}).get("selector_mode") == "ensemble"
    lines = [
        "# CUDA WFO Selector Report",
        "",
        f"- Profile: `{result['profile']}`",
        f"- CSV: `{result['csv']}`",
        f"- Decision: `{result['decision']['status']}`",
        f"- Fold mode: `{result['assumptions'].get('fold_mode', 'rolling')}`",
        f"- Folds: `{result['summary']['folds']}`",
        f"- Folds with train-only selection: `{result['summary']['folds_with_selection']}`",
        f"- Test pass ratio: `{result['summary']['test_pass_ratio']}`",
        f"- Aggregate return: `{result['aggregate']['return_pct']}%`",
        f"- Aggregate net PF: `{result['aggregate']['net_pf']}`",
        f"- Aggregate MDD: `{result['aggregate']['mdd_pct']}%`",
        f"- Aggregate trades: `{result['aggregate']['trades']}`",
        f"- DSR: `{result['dsr']['dsr']}`",
        f"- MC p05 return: `{result['monte_carlo'].get('return_pct_p05')}`",
        f"- Stress pass ratio: `{result['cost_stress']['pass_ratio']}`",
        "",
        "## Gates",
    ]
    for gate in result["gates"]:
        lines.append(f"- `{gate['name']}`: `{gate['pass']}` observed=`{gate['observed']}` threshold=`{gate['threshold']}`")
    lines.extend(["", "## Folds"])
    for row in result["folds"]:
        selected_ids = row.get("selected_param_ids")
        if is_ensemble and selected_ids:
            selected = "ENSEMBLE[" + "; ".join(str(item) for item in selected_ids[:3])
            if len(selected_ids) > 3:
                selected += f"; +{len(selected_ids) - 3} more"
            selected += "]"
        else:
            selected = row.get("selected_param_id") or "NONE"
        summary = row.get("test_summary", {})
        top_train = row.get("train_top_performance") or {}
        top_reasons = top_train.get("strict_reasons") or []
        failure_counts = row.get("train_strict_failure_counts_top_k") or {}
        failure_text = ";".join(f"{key}:{value}" for key, value in failure_counts.items()) if failure_counts else "-"
        diagnostics = (
            f", reason=`{row.get('reason')}`" if row.get("reason") else ""
        )
        diagnostics += (
            f", train_strict_top_k={row.get('train_strict_pass_count_top_k')}/{row.get('train_results_count')}, "
            f"train_top_return={top_train.get('return_pct')}%, train_top_pf={top_train.get('net_pf')}, "
            f"train_top_mdd={top_train.get('mdd_pct')}%, train_top_trades={top_train.get('trades')}, "
            f"train_top_strict={top_train.get('strict_pass')}, train_top_reasons=`{','.join(top_reasons) or '-'}`, "
            f"train_failures=`{failure_text}`"
            if row.get("train_results_count") is not None
            else ""
        )
        lines.append(
            f"- fold {row['fold']}: test {row['test_start']} to {row['test_end']}, "
            f"return={summary.get('return_pct')}%, pf={summary.get('net_pf')}, "
            f"mdd={summary.get('mdd_pct')}%, trades={summary.get('trades')}, "
            f"pass={row.get('test_pass')}, selected=`{selected}`{diagnostics}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selector(args: argparse.Namespace) -> dict[str, Any]:
    csv = _resolve(args.csv)
    fold_specs = _folds(
        args.start,
        args.end,
        args.train_months,
        args.test_months,
        args.step_months,
        args.purge_bars,
        args.embargo_bars,
        args.fold_mode,
    )
    if not fold_specs:
        raise SystemExit("No WFO folds generated.")

    fold_rows = []
    equity_curves = []
    trades_by_fold = []
    total_param_trials = 0
    for fold in fold_specs:
        print(
            f"[fold {fold['fold']}/{len(fold_specs)}] train {_fmt(fold['train_start'])} -> {_fmt(fold['train_end'])}; "
            f"test {_fmt(fold['test_start'])} -> {_fmt(fold['test_end'])}"
        )
        search_obj = _run_cuda_train_search(args, fold)
        total_param_trials = max(total_param_trials, int(search_obj.get("total_param_combinations") or 0))
        train_diagnostics = _train_search_diagnostics(search_obj)
        selection_limit = 1 if args.selector_mode == "best" else max(1, args.ensemble_size)
        selected_items = _select_candidates(search_obj, args.select_strict_only, selection_limit)
        if not selected_items:
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "train_start": _fmt(fold["train_start"]),
                    "train_end": _fmt(fold["train_end"]),
                    "test_start": _fmt(fold["test_start"]),
                    "test_end": _fmt(fold["test_end"]),
                    "selected_param_id": None,
                    "selected_parameters": None,
                    "test_summary": {"return_pct": 0.0, "net_pf": 0.0, "gross_pf": 0.0, "mdd_pct": 0.0, "trades": 0, "win_rate_pct": 0.0},
                    "test_pass": False,
                    "reason": "no_strict_train_candidate" if args.select_strict_only else "no_train_candidate",
                    **train_diagnostics,
                }
            )
            continue
        test_summary, equity, trades, component_summaries = _eval_selection(
            csv,
            fold,
            selected_items,
            args,
            commission=args.commission,
            slippage=args.slippage,
            funding=args.funding_rate_per_8h,
            funding_csv=args.funding_csv,
        )
        test_pass = _passes_test(test_summary, args)
        equity_curves.append(equity)
        trades_by_fold.append(trades)
        selected = selected_items[0]
        selected_param_id = selected.get("param_id") or selected.get("performance", {}).get("param_id")
        selected_param_ids = [
            item.get("param_id") or item.get("performance", {}).get("param_id")
            for item in selected_items
        ]
        selected_ensemble = [
            {
                "rank": int(item["rank"]),
                "param_id": item.get("param_id") or item.get("performance", {}).get("param_id"),
                "parameters": item["parameters"],
                "performance": item["performance"],
            }
            for item in selected_items
        ]
        fold_rows.append(
            {
                "fold": fold["fold"],
                "train_start": _fmt(fold["train_start"]),
                "train_end": _fmt(fold["train_end"]),
                "test_start": _fmt(fold["test_start"]),
                "test_end": _fmt(fold["test_end"]),
                "train_result_json": search_obj.get("_result_path"),
                "train_total_param_combinations": search_obj.get("total_param_combinations"),
                **train_diagnostics,
                "selected_rank": int(selected["rank"]),
                "selected_param_id": selected_param_id,
                "selected_param_ids": selected_param_ids,
                "selected_parameters": selected["parameters"],
                "selected_train_performance": selected["performance"],
                "selected_ensemble": selected_ensemble if args.selector_mode == "ensemble" else None,
                "component_summaries": component_summaries,
                "test_summary": test_summary,
                "test_pass": test_pass,
            }
        )

    selected_count = sum(1 for row in fold_rows if _selection_items(row))
    test_pass_count = sum(1 for row in fold_rows if row.get("test_pass"))
    aggregate = _aggregate(fold_rows, equity_curves, trades_by_fold, args.initial)
    combined_equity = _combine_equity(equity_curves, args.initial)
    dsr_trials = max(2, total_param_trials, selected_count)
    dsr = _deflated_sharpe(combined_equity, dsr_trials)
    mc = _monte_carlo(combined_equity, args.mc_runs, args.seed, args.initial)
    stress = _stress_selector(fold_rows, csv, args)

    test_pass_ratio = test_pass_count / len(fold_rows) if fold_rows else 0.0
    gates = [
        _gate("folds_with_selection", selected_count == len(fold_rows), selected_count, len(fold_rows), "Every fold must select a train-only candidate."),
        _gate("selector_test_pass_ratio", test_pass_ratio >= args.min_test_pass_ratio, round(test_pass_ratio, 4), f">= {args.min_test_pass_ratio}", "Forward-only test fold pass ratio."),
        _gate("aggregate_return_pct", aggregate["return_pct"] >= args.min_return, aggregate["return_pct"], f">= {args.min_return}", "Compounded OOS selector return."),
        _gate("aggregate_net_pf", _num(aggregate["net_pf"]) >= args.min_pf, aggregate["net_pf"], f">= {args.min_pf}", "Aggregate OOS net profit factor from selected test trades."),
        _gate("aggregate_mdd_pct", aggregate["mdd_pct"] <= args.max_mdd, aggregate["mdd_pct"], f"<= {args.max_mdd}", "Combined OOS selector max drawdown."),
        _gate("aggregate_trades", aggregate["trades"] >= args.min_trades, aggregate["trades"], f">= {args.min_trades}", "Aggregate selected OOS trades."),
        _gate("dsr", _num(dsr.get("dsr")) >= args.min_dsr, dsr.get("dsr"), f">= {args.min_dsr}", "Deflated Sharpe on combined forward-only OOS equity."),
        _gate("mc_prob_return_positive", _num(mc.get("prob_return_positive", 0.0)) >= args.min_mc_prob_positive, mc.get("prob_return_positive", 0.0), f">= {args.min_mc_prob_positive}", "Monte Carlo positive-return probability."),
        _gate("mc_return_p05", _num(mc.get("return_pct_p05", -float("inf"))) >= args.min_mc_return_p05, mc.get("return_pct_p05", -float("inf")), f">= {args.min_mc_return_p05}", "Monte Carlo 5th percentile return."),
        _gate("cost_stress_pass_ratio", stress["pass_ratio"] >= args.min_stress_pass_ratio, stress["pass_ratio"], f">= {args.min_stress_pass_ratio}", "Selected-parameter OOS cost stress pass ratio."),
    ]
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    decision = {
        "status": "READY_FOR_FORWARD_SHADOW_REVIEW" if not failed else "HOLD_AUTOMATED_PAPER",
        "ready_for_shadow": False,
        "ready_for_paper": False,
        "failed_gates": failed,
        "rationale": (
            "Forward-only selector gates passed, but this still requires explicit shadow-review promotion."
            if not failed
            else "Do not promote to shadow or paper; forward-only selector gates are not all satisfied."
        ),
    }

    return {
        "schema_version": 1,
        "profile": args.profile,
        "csv": str(csv),
        "period": {"start": args.start, "end": args.end},
        "assumptions": {
            "market": "Binance USD-M perpetual",
            "commission_rate": args.commission,
            "slippage_rate": args.slippage,
            "funding_rate_per_8h": args.funding_rate_per_8h,
            "funding_model": "actual_funding_events" if args.funding_csv is not None else "constant_per_8h",
            "funding_rate_csv": str(args.funding_csv) if args.funding_csv is not None else None,
            "entry_delay_bars": args.entry_delay_bars,
            "fold_mode": args.fold_mode,
            "fold_schedule": {
                "train_months": args.train_months,
                "test_months": args.test_months,
                "step_months": args.step_months,
                "purge_bars": args.purge_bars,
                "embargo_bars": args.embargo_bars,
            },
            "selection": (
                "CUDA search on train fold only; top-N equal-weight ensemble CPU reference evaluation on unseen test fold."
                if args.selector_mode == "ensemble"
                else "CUDA search on train fold only; CPU reference evaluation on unseen test fold."
            ),
            "selector_mode": args.selector_mode,
            "ensemble_size": args.ensemble_size if args.selector_mode == "ensemble" else 1,
            "test_warmup": "Indicators are computed inside each test slice, matching existing fixed-candidate WFO conservatism.",
        },
        "criteria": {
            "min_return": args.min_return,
            "min_pf": args.min_pf,
            "max_mdd": args.max_mdd,
            "min_trades": args.min_trades,
            "test_min_return": args.test_min_return,
            "test_min_pf": args.test_min_pf,
            "test_max_mdd": args.test_max_mdd,
            "test_min_trades": args.test_min_trades,
            "min_test_pass_ratio": args.min_test_pass_ratio,
            "min_dsr": args.min_dsr,
            "min_mc_prob_positive": args.min_mc_prob_positive,
            "min_mc_return_p05": args.min_mc_return_p05,
            "min_stress_pass_ratio": args.min_stress_pass_ratio,
        },
        "summary": {
            "folds": len(fold_rows),
            "folds_with_selection": selected_count,
            "test_pass_count": test_pass_count,
            "test_pass_ratio": round(test_pass_ratio, 4),
            "dsr_trials": dsr_trials,
        },
        "aggregate": aggregate,
        "dsr": dsr,
        "monte_carlo": mc,
        "cost_stress": stress,
        "gates": gates,
        "decision": decision,
        "folds": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward-only WFO selector: train-fold CUDA search, test-fold CPU reference.")
    parser.add_argument("--profile", default="phase17_long_breakout_regime")
    parser.add_argument("--csv", type=Path, default=Path("wfa_optimized_params_output/live_shadow_phase14_data/BTCUSDT_1h.csv"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument(
        "--fold-mode",
        choices=["rolling", "anchored"],
        default="rolling",
        help="rolling keeps a fixed-length train window; anchored expands train from the initial start date.",
    )
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0001)
    parser.add_argument("--funding-csv", type=Path, default=None, help="Optional Binance funding-rate event CSV for actual funding charges.")
    parser.add_argument("--rank-metric", default="robust", choices=["return", "strict_return", "robust", "smooth", "dense"])
    parser.add_argument("--select-strict-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector-mode", choices=["best", "ensemble"], default="best")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--train-min-return", type=float, default=30.0)
    parser.add_argument("--train-min-pf", type=float, default=1.3)
    parser.add_argument("--train-max-mdd", type=float, default=25.0)
    parser.add_argument("--train-min-trades", type=int, default=30)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--test-min-return", type=float, default=0.0)
    parser.add_argument("--test-min-pf", type=float, default=1.0)
    parser.add_argument("--test-max-mdd", type=float, default=25.0)
    parser.add_argument("--test-min-trades", type=int, default=3)
    parser.add_argument("--min-test-pass-ratio", type=float, default=0.6)
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-mc-prob-positive", type=float, default=0.6)
    parser.add_argument("--min-mc-return-p05", type=float, default=0.0)
    parser.add_argument("--stress-commission-mult", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--stress-slippage-mult", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--stress-funding-mult", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--min-stress-pass-ratio", type=float, default=0.5)
    parser.add_argument("--timeout-minutes", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("wfa_optimized_params_output"))
    parser.add_argument("--out", type=Path, default=Path("wfa_optimized_params_output/wfo_selector_phase17_long_breakout_2019_2026.json"))
    parser.add_argument("--out-md", type=Path, default=Path("wfa_optimized_params_output/wfo_selector_phase17_long_breakout_2019_2026.md"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    args.csv = _resolve(args.csv)
    args.funding_csv = _resolve(args.funding_csv) if args.funding_csv is not None else None
    result = run_selector(args)
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    out_md = _resolve(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(out_md, result)
    print(f"wrote {out}")
    print(f"wrote {out_md}")
    print(
        "selector summary "
        f"decision={result['decision']['status']} "
        f"return={result['aggregate']['return_pct']} "
        f"pf={result['aggregate']['net_pf']} "
        f"mdd={result['aggregate']['mdd_pct']} "
        f"test_pass_ratio={result['summary']['test_pass_ratio']} "
        f"failed={result['decision']['failed_gates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
