from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from analyze_dsr_return_shape import (  # noqa: E402
    _combine_equity,
    _combine_weighted_equity,
    _json_default,
    _num,
    _profit_factor,
    _required_annualized_sharpe,
    _return_distribution,
    _trade_distribution,
)
from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative  # noqa: E402
from validate_v2_strategy import (  # noqa: E402
    _deflated_sharpe,
    _equity_returns,
    _max_drawdown_from_equity,
    _monte_carlo,
)


def _resolve(path: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def _parse_weights(raw: str | None, count: int) -> np.ndarray:
    if count <= 0:
        raise SystemExit("At least one --wfo-json is required.")
    if raw is None:
        weights = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
        if len(values) != count:
            raise SystemExit(f"--weights count ({len(values)}) must match --wfo-json count ({count}).")
        weights = np.array(values, dtype=np.float64)
        total = float(np.sum(weights))
        if total <= 0.0:
            raise SystemExit("--weights must sum to a positive value.")
        weights = weights / total
    return weights


def _weights_label(weights: np.ndarray) -> str:
    return ",".join(f"{float(value):.4f}".rstrip("0").rstrip(".") for value in weights)


def _weight_grid(count: int, step: float, min_weight: float, max_weight: float) -> np.ndarray:
    if count <= 0:
        raise SystemExit("At least one source is required for the weight grid.")
    if step <= 0.0 or step > 1.0:
        raise SystemExit("--weight-grid-step must be in (0, 1].")
    units = int(round(1.0 / step))
    if abs((units * step) - 1.0) > 1e-9:
        raise SystemExit("--weight-grid-step must divide 1.0 exactly, e.g. 0.1 or 0.05.")
    min_units = int(math.ceil(min_weight / step - 1e-9))
    max_units = int(math.floor(max_weight / step + 1e-9))
    rows: list[list[float]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            if min_units <= remaining <= max_units:
                rows.append([*(value * step for value in prefix), remaining * step])
            return
        lo = min_units
        hi = min(max_units, remaining - min_units * (slots - 1))
        for value in range(lo, hi + 1):
            rec([*prefix, value], remaining - value, slots - 1)

    rec([], units, count)
    if not rows:
        raise SystemExit("Weight grid is empty. Relax min/max weights or use a coarser step.")
    return np.array(rows, dtype=np.float64)


def _float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _eval_params_with_costs(
    csv: Path,
    start: str,
    end: str,
    params: dict[str, Any],
    assumptions: dict[str, Any],
    initial: float,
    *,
    commission_mult: float,
    slippage_mult: float,
    funding_mult: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    df = _load_ohlcv(csv, start, end)
    funding_csv = _resolve(assumptions.get("funding_rate_csv"))
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    if funding_cumulative is not None:
        funding_cumulative = funding_cumulative * float(funding_mult)
    _, equity, trades = cpu_reference_backtest(
        df,
        params,
        initial_balance=initial,
        commission_rate=_num(assumptions.get("commission_rate")) * float(commission_mult),
        slippage_rate=_num(assumptions.get("slippage_rate")) * float(slippage_mult),
        entry_delay_bars=int(assumptions.get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h")) * float(funding_mult),
        include_equity=True,
        include_trades=True,
        funding_cumulative=funding_cumulative,
    )
    return equity, trades


def _load_wfo_equity(
    path: Path,
    initial: float,
    *,
    commission_mult: float = 1.0,
    slippage_mult: float = 1.0,
    funding_mult: float = 1.0,
) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    csv = _resolve(obj["csv"])
    assumptions = obj.get("assumptions", {})
    fold_equities: list[np.ndarray] = []
    all_trades: list[dict[str, Any]] = []

    for fold in obj.get("folds", []):
        if fold.get("selected_ensemble"):
            selections = list(fold["selected_ensemble"])
        elif fold.get("selected_parameters"):
            selections = [{"parameters": fold["selected_parameters"]}]
        else:
            selections = []

        if not selections:
            fold_equities.append(np.array([initial], dtype=np.float64))
            continue

        equities = []
        selection_weights = np.full(len(selections), 1.0 / len(selections), dtype=np.float64)
        for idx, selection in enumerate(selections):
            equity, trades = _eval_params_with_costs(
                csv,
                fold["test_start"],
                fold["test_end"],
                selection["parameters"],
                assumptions,
                initial,
                commission_mult=commission_mult,
                slippage_mult=slippage_mult,
                funding_mult=funding_mult,
            )
            equities.append(equity)
            for trade in trades:
                scaled = dict(trade)
                for key in ("gross_pnl", "net_pnl", "commission", "slippage_cost", "funding"):
                    if key in scaled:
                        scaled[key] = _num(scaled[key]) * float(selection_weights[idx])
                all_trades.append(scaled)

        fold_equities.append(_combine_weighted_equity(equities, selection_weights, initial))

    combined = _combine_equity(fold_equities, initial)
    return {
        "path": str(path),
        "profile": obj.get("profile", path.stem),
        "selector_mode": obj.get("assumptions", {}).get("selector_mode"),
        "fold_mode": obj.get("assumptions", {}).get("fold_mode"),
        "summary": obj.get("summary", {}),
        "aggregate": obj.get("aggregate", {}),
        "dsr": obj.get("dsr", {}),
        "decision": obj.get("decision", {}),
        "equity": combined,
        "trades": all_trades,
        "dsr_trials": int(obj.get("summary", {}).get("dsr_trials", 2)),
    }


def _combine_portfolio(
    equity_curves: list[np.ndarray],
    trades_by_source: list[list[dict[str, Any]]],
    weights: np.ndarray,
    initial: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    combined = _combine_weighted_equity(equity_curves, weights, initial)
    weighted_trades: list[dict[str, Any]] = []
    for source_trades, weight in zip(trades_by_source, weights):
        for trade in source_trades:
            scaled = dict(trade)
            for key in ("gross_pnl", "net_pnl", "commission", "slippage_cost", "funding"):
                if key in scaled:
                    scaled[key] = _num(scaled[key]) * float(weight)
            weighted_trades.append(scaled)
    return combined, weighted_trades


def _decision(gates: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [gate for gate in gates if not gate["pass"]]
    return {
        "status": "PORTFOLIO_DIAGNOSTIC_REVIEW" if not failed else "HOLD_AUTOMATED_PAPER",
        "ready_for_shadow": False,
        "ready_for_paper": False,
        "failed_gates": [gate["name"] for gate in failed],
        "rationale": (
            "Portfolio diagnostic metrics passed, but shadow/paper still require a formal export path."
            if not failed
            else "Portfolio is not eligible for automated paper trading or shadow promotion until failed gates pass."
        ),
    }


def _gate(name: str, passed: bool, observed: Any, threshold: Any, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _portfolio_shape(
    label: str,
    equity_curves: list[np.ndarray],
    trades_by_source: list[list[dict[str, Any]]],
    weights: np.ndarray,
    n_trials: int,
    initial: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined, weighted_trades = _combine_portfolio(equity_curves, trades_by_source, weights, initial)
    final_balance = float(combined[-1]) if combined.size else initial
    dsr = _deflated_sharpe(combined, n_trials)
    returns = _equity_returns(combined)
    pnl = np.array([_num(row.get("net_pnl")) for row in weighted_trades], dtype=np.float64)
    summary = {
        "final_balance": round(final_balance, 6),
        "return_pct": round((final_balance - initial) / initial * 100.0, 6) if initial else 0.0,
        "mdd_pct": round(_max_drawdown_from_equity(combined), 6),
        "trades": len(weighted_trades),
        "trade_profit_factor": _profit_factor(pnl) if pnl.size else 0.0,
    }
    monte_carlo = _monte_carlo(combined, args.mc_runs, args.seed, initial)
    gates = [
        _gate("portfolio_return_pct", summary["return_pct"] >= args.min_return, summary["return_pct"], f">= {args.min_return}", "Combined WFO portfolio return."),
        _gate("portfolio_trade_pf", _num(summary["trade_profit_factor"]) >= args.min_pf, summary["trade_profit_factor"], f">= {args.min_pf}", "Trade-level net PF after weights."),
        _gate("portfolio_mdd_pct", summary["mdd_pct"] <= args.max_mdd, summary["mdd_pct"], f"<= {args.max_mdd}", "Combined WFO portfolio drawdown."),
        _gate("portfolio_trades", summary["trades"] >= args.min_trades, summary["trades"], f">= {args.min_trades}", "Combined weighted source trades."),
        _gate("portfolio_dsr", dsr.get("dsr", 0.0) >= args.min_dsr, dsr.get("dsr"), f">= {args.min_dsr}", "Deflated Sharpe on the combined WFO equity stream."),
        _gate("portfolio_mc_prob_positive", monte_carlo.get("prob_return_positive", 0.0) >= args.min_mc_prob_positive, monte_carlo.get("prob_return_positive"), f">= {args.min_mc_prob_positive}", "Monte Carlo positive-return probability."),
        _gate("portfolio_mc_return_p05", monte_carlo.get("return_pct_p05", -1e9) >= args.min_mc_return_p05, monte_carlo.get("return_pct_p05"), f">= {args.min_mc_return_p05}", "Monte Carlo 5th percentile return."),
    ]
    return {
        "label": label,
        "summary": summary,
        "dsr": dsr,
        "monte_carlo": monte_carlo,
        "gates_without_formal_pbo": gates,
        "decision_without_formal_pbo": _decision(gates),
        "required_annualized_sharpe": {
            "dsr_0_80": _required_annualized_sharpe(dsr, 0.80),
            "dsr_0_95": _required_annualized_sharpe(dsr, 0.95),
        },
        "return_distribution": _return_distribution(returns),
        "trade_distribution": _trade_distribution(weighted_trades),
    }


def _portfolio_equity_from_source_curves(
    equity_curves: list[np.ndarray],
    weights: np.ndarray,
    start_idx: int,
    end_idx: int,
    initial: float,
) -> np.ndarray:
    if end_idx <= start_idx:
        return np.array([initial], dtype=np.float64)
    segments = []
    for curve in equity_curves:
        segment = curve[start_idx:end_idx]
        if segment.size == 0:
            segments.append(np.array([initial], dtype=np.float64))
            continue
        base = float(segment[0])
        if base == 0.0:
            normalized = np.ones(segment.size, dtype=np.float64)
        else:
            normalized = segment / base
        segments.append(normalized * initial)
    return _combine_weighted_equity(segments, weights, initial)


def _score_equity(equity: np.ndarray, *, metric: str = "robust_return", n_trials: int = 2) -> dict[str, Any]:
    if equity.size < 2:
        return {"score": -1e9, "return_pct": 0.0, "return_pf": 0.0, "mdd_pct": 0.0, "observations": 0}
    returns = _equity_returns(equity)
    gains = float(returns[returns > 0.0].sum())
    losses = float(-returns[returns < 0.0].sum())
    if losses == 0.0:
        return_pf = 10.0 if gains > 0.0 else 0.0
    else:
        return_pf = gains / losses
    final_balance = float(equity[-1])
    initial = float(equity[0])
    return_pct = (final_balance - initial) / initial * 100.0 if initial else 0.0
    mdd_pct = _max_drawdown_from_equity(equity)
    dsr = _deflated_sharpe(equity, n_trials)
    if metric == "robust_return":
        score = return_pct + 10.0 * math.log(max(0.01, return_pf)) - 0.25 * mdd_pct
    elif metric == "sharpe":
        score = float(dsr.get("annualized_sharpe", 0.0)) + 0.02 * return_pct - 0.02 * mdd_pct
    elif metric == "dsr":
        score = 100.0 * float(dsr.get("dsr", 0.0)) + 0.05 * return_pct - 0.05 * mdd_pct
    else:
        raise SystemExit(f"Unknown portfolio score metric: {metric}")
    return {
        "score": round(float(score), 8),
        "score_metric": metric,
        "return_pct": round(float(return_pct), 6),
        "return_pf": round(float(return_pf), 6),
        "mdd_pct": round(float(mdd_pct), 6),
        "annualized_sharpe": dsr.get("annualized_sharpe"),
        "dsr": dsr.get("dsr"),
        "observations": int(returns.size),
    }


def _score_weights_on_interval(
    equity_curves: list[np.ndarray],
    weights: np.ndarray,
    start_idx: int,
    end_idx: int,
    initial: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return _score_equity(
        _portfolio_equity_from_source_curves(equity_curves, weights, start_idx, end_idx, initial),
        metric=args.portfolio_score_metric,
        n_trials=int(args.n_trials) if args.n_trials is not None else 2,
    )


def _portfolio_weight_pbo(
    equity_curves: list[np.ndarray],
    weight_candidates: np.ndarray,
    args: argparse.Namespace,
    initial: float,
) -> dict[str, Any]:
    splits = int(args.portfolio_pbo_splits)
    if splits < 4 or splits % 2 != 0 or len(weight_candidates) < 2:
        return {"enabled": False, "reason": "requires even splits >=4 and at least two weight candidates"}
    min_len = min((curve.size for curve in equity_curves), default=0)
    if min_len < splits * max(2, int(args.min_pbo_block_rows)):
        return {"enabled": False, "reason": "not enough aligned equity observations"}
    block_edges = np.linspace(0, min_len, splits + 1, dtype=int)
    left_trim = max(0, int(args.embargo_bars))
    right_trim = max(0, int(args.purge_bars))
    intervals: list[tuple[int, int]] = []
    raw_rows = []
    effective_rows = []
    for idx in range(splits):
        raw_start = int(block_edges[idx])
        raw_end = int(block_edges[idx + 1])
        raw_rows.append(raw_end - raw_start)
        start = min(raw_end, raw_start + left_trim)
        end = max(start, raw_end - right_trim)
        intervals.append((start, end))
        effective_rows.append(max(0, end - start))

    scores = np.full((len(weight_candidates), splits), -1e9, dtype=np.float64)
    detail = [[{} for _ in range(splits)] for _ in range(len(weight_candidates))]
    for c_idx, weights in enumerate(weight_candidates):
        for b_idx, (start, end) in enumerate(intervals):
            if end - start < int(args.min_pbo_block_rows):
                continue
            row = _score_weights_on_interval(equity_curves, weights, start, end, initial, args)
            scores[c_idx, b_idx] = float(row["score"])
            detail[c_idx][b_idx] = row

    lambdas = []
    ranks = []
    winners = []
    block_ids = range(splits)
    for train_blocks in itertools.combinations(block_ids, splits // 2):
        test_blocks = [idx for idx in block_ids if idx not in set(train_blocks)]
        train_scores = np.nanmean(scores[:, list(train_blocks)], axis=1)
        test_scores = np.nanmean(scores[:, test_blocks], axis=1)
        if not np.isfinite(train_scores).any() or not np.isfinite(test_scores).any():
            continue
        winner = int(np.nanargmax(train_scores))
        order = np.argsort(np.argsort(test_scores))
        pct_rank = float(order[winner] / max(1, len(weight_candidates) - 1))
        pct_rank = min(1.0 - 1e-9, max(1e-9, pct_rank))
        ranks.append(pct_rank)
        lambdas.append(math.log(pct_rank / (1.0 - pct_rank)))
        winners.append(
            {
                "train_blocks": list(train_blocks),
                "test_blocks": test_blocks,
                "winner_index": winner,
                "winner_weights": [round(float(value), 6) for value in weight_candidates[winner]],
                "test_percentile": round(pct_rank, 6),
                "train_score": round(float(train_scores[winner]), 8),
                "test_score": round(float(test_scores[winner]), 8),
            }
        )
    if not lambdas:
        return {"enabled": False, "reason": "no finite samples"}

    full_scores = []
    for c_idx, weights in enumerate(weight_candidates):
        full_equity = _portfolio_equity_from_source_curves(equity_curves, weights, 0, min_len, initial)
        row = _score_equity(
            full_equity,
            metric=args.portfolio_score_metric,
            n_trials=int(args.n_trials) if args.n_trials is not None else 2,
        )
        full_scores.append((float(row["score"]), c_idx, row))
    full_scores.sort(reverse=True, key=lambda item: item[0])
    top = []
    for score, idx, row in full_scores[: int(args.portfolio_pbo_top_show)]:
        top.append(
            {
                "index": idx,
                "weights": [round(float(value), 6) for value in weight_candidates[idx]],
                "label": _weights_label(weight_candidates[idx]),
                "score": round(score, 8),
                "summary": row,
            }
        )
    lambdas_arr = np.array(lambdas, dtype=np.float64)
    ranks_arr = np.array(ranks, dtype=np.float64)
    return {
        "enabled": True,
        "note": "Purged/embargoed CSCV/PBO over portfolio weight-grid candidates using WFO equity curves. Weight selection is diagnostic and does not export a tradable portfolio.",
        "splits": splits,
        "weight_candidates": int(len(weight_candidates)),
        "weight_grid_step": args.weight_grid_step,
        "score_metric": args.portfolio_score_metric,
        "min_weight": args.weight_grid_min,
        "max_weight": args.weight_grid_max,
        "purge_bars": right_trim,
        "embargo_bars": left_trim,
        "raw_block_rows_min": int(min(raw_rows)) if raw_rows else 0,
        "effective_block_rows_min": int(min(effective_rows)) if effective_rows else 0,
        "effective_block_rows": effective_rows,
        "samples": len(lambdas),
        "pbo": round(float(np.mean(lambdas_arr <= 0.0)), 6),
        "median_test_percentile": round(float(np.median(ranks_arr)), 6),
        "mean_test_percentile": round(float(np.mean(ranks_arr)), 6),
        "top_full_sample_weights": top,
        "winner_samples": winners[: int(args.portfolio_pbo_top_show)],
    }


def _portfolio_meta_selection(
    equity_curves: list[np.ndarray],
    weight_candidates: np.ndarray,
    args: argparse.Namespace,
    initial: float,
) -> dict[str, Any]:
    splits = int(args.meta_splits)
    train_blocks = int(args.meta_train_blocks)
    if splits < 3 or train_blocks < 1 or train_blocks >= splits:
        return {"enabled": False, "reason": "requires meta_splits >=3 and 1 <= meta_train_blocks < meta_splits"}
    min_len = min((curve.size for curve in equity_curves), default=0)
    if min_len < splits * 2:
        return {"enabled": False, "reason": "not enough aligned equity observations"}
    block_edges = np.linspace(0, min_len, splits + 1, dtype=int)
    selected_rows = []
    test_equities = []
    for test_block in range(train_blocks, splits):
        train_ids = list(range(test_block - train_blocks, test_block))
        train_scores = np.full(len(weight_candidates), -1e9, dtype=np.float64)
        for c_idx, weights in enumerate(weight_candidates):
            block_scores = []
            for block_id in train_ids:
                row = _score_weights_on_interval(
                    equity_curves,
                    weights,
                    int(block_edges[block_id]),
                    int(block_edges[block_id + 1]),
                    initial,
                    args,
                )
                block_scores.append(float(row["score"]))
            train_scores[c_idx] = float(np.mean(block_scores)) if block_scores else -1e9
        winner = int(np.argmax(train_scores))
        start = int(block_edges[test_block])
        end = int(block_edges[test_block + 1])
        test_equity = _portfolio_equity_from_source_curves(equity_curves, weight_candidates[winner], start, end, initial)
        test_equities.append(test_equity)
        selected_rows.append(
            {
                "test_block": test_block,
                "train_blocks": train_ids,
                "selected_index": winner,
                "selected_weights": [round(float(value), 6) for value in weight_candidates[winner]],
                "train_score": round(float(train_scores[winner]), 8),
                "test_summary": _score_equity(
                    test_equity,
                    metric=args.portfolio_score_metric,
                    n_trials=int(args.n_trials) if args.n_trials is not None else 2,
                ),
            }
        )
    combined = _combine_equity(test_equities, initial)
    dsr = _deflated_sharpe(combined, int(args.n_trials) if args.n_trials is not None else max(2, len(weight_candidates)))
    mc = _monte_carlo(combined, args.mc_runs, args.seed, initial)
    final_balance = float(combined[-1]) if combined.size else initial
    summary = {
        "final_balance": round(final_balance, 6),
        "return_pct": round((final_balance - initial) / initial * 100.0, 6) if initial else 0.0,
        "mdd_pct": round(_max_drawdown_from_equity(combined), 6),
        "observations": int(max(0, combined.size - 1)),
        "dsr": dsr,
        "monte_carlo": mc,
    }
    gates = [
        _gate("meta_return_pct", summary["return_pct"] >= args.min_return, summary["return_pct"], f">= {args.min_return}", "Chronological meta-selected portfolio return."),
        _gate("meta_mdd_pct", summary["mdd_pct"] <= args.max_mdd, summary["mdd_pct"], f"<= {args.max_mdd}", "Chronological meta-selected portfolio drawdown."),
        _gate("meta_dsr", dsr.get("dsr", 0.0) >= args.min_dsr, dsr.get("dsr"), f">= {args.min_dsr}", "Deflated Sharpe on chronological meta-selected portfolio."),
        _gate("meta_mc_return_p05", mc.get("return_pct_p05", -1e9) >= args.min_mc_return_p05, mc.get("return_pct_p05"), f">= {args.min_mc_return_p05}", "Meta-selected Monte Carlo 5th percentile return."),
    ]
    return {
        "enabled": True,
        "note": "Chronological rolling meta-selection over portfolio weight-grid candidates. This is a diagnostic approximation over concatenated WFO equity curves.",
        "splits": splits,
        "train_blocks": train_blocks,
        "test_blocks": splits - train_blocks,
        "weight_candidates": int(len(weight_candidates)),
        "summary": summary,
        "gates": gates,
        "decision": _decision(gates),
        "selected": selected_rows,
    }


def _pairwise_return_corr(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a = _equity_returns(sources[i]["equity"])
            b = _equity_returns(sources[j]["equity"])
            n = min(a.size, b.size)
            if n < 3:
                corr = None
            else:
                corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
            rows.append(
                {
                    "left": Path(sources[i]["path"]).stem,
                    "right": Path(sources[j]["path"]).stem,
                    "observations": int(n),
                    "return_corr": None if corr is None else round(corr, 6),
                }
            )
    return rows


def _cost_stress(args: argparse.Namespace, weights: np.ndarray, n_trials: int, initial: float) -> dict[str, Any]:
    rows = []
    for commission_mult in _float_list(args.stress_commission_mult):
        for slippage_mult in _float_list(args.stress_slippage_mult):
            for funding_mult in _float_list(args.stress_funding_mult):
                sources = [
                    _load_wfo_equity(
                        _resolve(path),
                        initial,
                        commission_mult=commission_mult,
                        slippage_mult=slippage_mult,
                        funding_mult=funding_mult,
                    )
                    for path in args.wfo_json
                ]
                shape = _portfolio_shape(
                    args.label,
                    [source["equity"] for source in sources],
                    [source["trades"] for source in sources],
                    weights,
                    n_trials,
                    initial,
                    args,
                )
                summary = shape["summary"]
                passed = (
                    summary["return_pct"] >= args.min_return
                    and _num(summary["trade_profit_factor"]) >= args.min_pf
                    and summary["mdd_pct"] <= args.max_mdd
                    and summary["trades"] >= args.min_trades
                )
                rows.append(
                    {
                        "commission_mult": commission_mult,
                        "slippage_mult": slippage_mult,
                        "funding_mult": funding_mult,
                        "summary": summary,
                        "pass": bool(passed),
                    }
                )
    return {
        "scenarios": rows,
        "pass_ratio": round(sum(1 for row in rows if row["pass"]) / len(rows), 6) if rows else 0.0,
    }


def _source_rows(sources: list[dict[str, Any]], weights: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for source, weight in zip(sources, weights):
        agg = source.get("aggregate", {})
        dsr = source.get("dsr", {})
        rows.append(
            {
                "weight": round(float(weight), 6),
                "file": Path(source["path"]).name,
                "profile": source.get("profile"),
                "selector_mode": source.get("selector_mode"),
                "fold_mode": source.get("fold_mode"),
                "return_pct": agg.get("return_pct"),
                "net_pf": agg.get("net_pf"),
                "mdd_pct": agg.get("mdd_pct"),
                "trades": agg.get("trades"),
                "dsr": dsr.get("dsr"),
                "annualized_sharpe": dsr.get("annualized_sharpe"),
                "failed_gates": source.get("decision", {}).get("failed_gates", []),
            }
        )
    return rows


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    shape = result["portfolio"]
    summary = shape["summary"]
    dsr = shape["dsr"]
    ret = shape["return_distribution"]
    trade = shape["trade_distribution"]
    mc = shape.get("monte_carlo", {})
    stress = result.get("cost_stress", {})
    pbo = result.get("portfolio_weight_pbo", {})
    meta = result.get("portfolio_meta_selection", {})
    lines = [
        "# WFO Portfolio Diagnostic",
        "",
        "This is a research diagnostic only. It combines already generated WFO equity curves by fixed capital weights and does not replace formal portfolio PBO/export gates.",
        "",
        "## Portfolio",
        "",
        "| return | mdd | trades | trade_pf | ann_sharpe | dsr | mc_p05 | stress_pass | req95 | zero_ratio | kurtosis | max_loss_run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {summary['return_pct']} | {summary['mdd_pct']} | {summary['trades']} | "
            f"{summary['trade_profit_factor']} | {dsr.get('annualized_sharpe')} | {dsr.get('dsr')} | "
            f"{mc.get('return_pct_p05')} | {stress.get('pass_ratio')} | "
            f"{shape['required_annualized_sharpe'].get('dsr_0_95')} | {ret.get('zero_return_ratio')} | "
            f"{dsr.get('kurtosis')} | {trade.get('max_consecutive_losses')} |"
        ),
        "",
        "## Formal Diagnostics",
        "",
        "| pbo | median_test_pct | weight_candidates | meta_return | meta_mdd | meta_dsr | formal_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {pbo.get('pbo')} | {pbo.get('median_test_percentile')} | {pbo.get('weight_candidates')} | "
            f"{meta.get('summary', {}).get('return_pct')} | {meta.get('summary', {}).get('mdd_pct')} | "
            f"{meta.get('summary', {}).get('dsr', {}).get('dsr')} | {result.get('formal_promotion_gate', {}).get('status')} |"
        ),
        "",
        "## Sources",
        "",
        "| weight | file | return | pf | mdd | trades | dsr | failed_gates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["sources"]:
        failed = ",".join(row.get("failed_gates", []))
        lines.append(
            f"| {row['weight']} | {row['file']} | {row.get('return_pct')} | {row.get('net_pf')} | "
            f"{row.get('mdd_pct')} | {row.get('trades')} | {row.get('dsr')} | {failed} |"
        )
    lines.extend(["", "## Pairwise Return Correlation", "", "| left | right | corr | observations |", "| --- | --- | --- | --- |"])
    for row in result["pairwise_return_corr"]:
        lines.append(f"| {row['left']} | {row['right']} | {row['return_corr']} | {row['observations']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine WFO selector equity curves as a fixed-weight diagnostic portfolio.")
    parser.add_argument("--wfo-json", action="append", type=Path, required=True)
    parser.add_argument("--weights", default=None, help="Comma-separated capital weights; normalized automatically.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override DSR trial count for the combined diagnostic.")
    parser.add_argument("--label", default="wfo_portfolio")
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--min-mc-prob-positive", type=float, default=0.6)
    parser.add_argument("--min-mc-return-p05", type=float, default=0.0)
    parser.add_argument("--min-stress-pass-ratio", type=float, default=0.5)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--stress-commission-mult", default="1.0,1.5,2.0")
    parser.add_argument("--stress-slippage-mult", default="1.0,2.0")
    parser.add_argument("--stress-funding-mult", default="0.0,1.0,2.0")
    parser.add_argument("--portfolio-pbo-splits", type=int, default=6)
    parser.add_argument("--portfolio-pbo-top-show", type=int, default=10)
    parser.add_argument("--portfolio-score-metric", choices=["robust_return", "sharpe", "dsr"], default="robust_return")
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument("--weight-grid-min", type=float, default=0.0)
    parser.add_argument("--weight-grid-max", type=float, default=1.0)
    parser.add_argument("--min-pbo-block-rows", type=int, default=500)
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--max-pbo", type=float, default=0.5)
    parser.add_argument("--meta-splits", type=int, default=8)
    parser.add_argument("--meta-train-blocks", type=int, default=4)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    initial = 10_000.0
    weights = _parse_weights(args.weights, len(args.wfo_json))
    sources = [_load_wfo_equity(_resolve(path), initial) for path in args.wfo_json]
    n_trials = int(args.n_trials) if args.n_trials is not None else int(sum(source["dsr_trials"] for source in sources))
    result = {
        "schema_version": 1,
        "label": args.label,
        "assumptions": {
            "capital_allocation": "fixed_weight_split",
            "equity_alignment": "diagnostic_min_length_index_alignment",
            "n_trials_basis": "manual_override" if args.n_trials is not None else "sum_source_wfo_dsr_trials",
            "n_trials": n_trials,
            "promotion_policy": "diagnostic_only_not_shadow_or_paper_eligible_without_formal_portfolio_pbo_export",
        },
        "sources": _source_rows(sources, weights),
        "pairwise_return_corr": _pairwise_return_corr(sources),
        "portfolio": _portfolio_shape(
            args.label,
            [source["equity"] for source in sources],
            [source["trades"] for source in sources],
            weights,
            n_trials,
            initial,
            args,
        ),
    }
    stress = _cost_stress(args, weights, n_trials, initial)
    result["cost_stress"] = stress
    weight_candidates = _weight_grid(
        len(args.wfo_json),
        args.weight_grid_step,
        args.weight_grid_min,
        args.weight_grid_max,
    )
    pbo = _portfolio_weight_pbo([source["equity"] for source in sources], weight_candidates, args, initial)
    meta = _portfolio_meta_selection([source["equity"] for source in sources], weight_candidates, args, initial)
    result["portfolio_weight_pbo"] = pbo
    result["portfolio_meta_selection"] = meta
    pbo_gate_pass = bool(pbo.get("enabled")) and _num(pbo.get("pbo"), 1.0) <= args.max_pbo
    meta_failed = meta.get("decision", {}).get("failed_gates", ["portfolio_meta_selection_missing"])
    meta_gate_pass = bool(meta.get("enabled")) and not meta_failed
    diagnostic_gates = result["portfolio"].get("gates_without_formal_pbo", [])
    diagnostic_pass = not [gate for gate in diagnostic_gates if not gate.get("pass")]
    stress_pass = stress["pass_ratio"] >= args.min_stress_pass_ratio
    result["formal_promotion_gate"] = {
        "status": "HOLD_AUTOMATED_PAPER",
        "ready_for_shadow": False,
        "ready_for_paper": False,
        "failed_gates": [
            *([] if diagnostic_pass else ["portfolio_diagnostic_gates"]),
            *([] if pbo_gate_pass else ["formal_portfolio_pbo"]),
            *([] if meta_gate_pass else ["formal_portfolio_meta_selection"]),
            *([] if stress_pass else ["portfolio_cost_stress"]),
            "formal_portfolio_export",
        ],
        "pbo_gate_pass": pbo_gate_pass,
        "meta_gate_pass": meta_gate_pass,
        "stress_pass_ratio": stress["pass_ratio"],
        "stress_gate_pass": stress_pass,
        "rationale": "Portfolio diagnostics are encouraging, but shadow/paper stays disabled until every formal portfolio gate passes and an export path exists.",
    }

    text = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    if args.out_json:
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
    if args.out_md:
        out_md = _resolve(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, result)
    if args.out_json or args.out_md:
        print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
