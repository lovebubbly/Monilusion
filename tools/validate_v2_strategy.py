from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.backtest_v2 import coarse_grid, dense_grid, load_csv, simulate


@dataclass(frozen=True)
class Criteria:
    min_return_pct: float
    min_pf: float
    max_mdd_pct: float
    min_trades: int
    min_wfo_pass_ratio: float
    max_pbo: float
    min_dsr: float
    min_mc_prob_positive: float
    min_mc_return_p05: float
    min_stress_pass_ratio: float


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def _num(value: Any, default: float = 0.0) -> float:
    if value == "inf":
        return float("inf")
    if value is None:
        return default
    return float(value)


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": round(_num(report.get("Total Net Pnl Percentage")), 4),
        "net_pf": round(_num(report.get("Net Profit Factor", report.get("Profit Factor"))), 4),
        "gross_pf": round(_num(report.get("Gross Profit Factor")), 4),
        "mdd_pct": round(_num(report.get("Max Drawdown Percentage")), 4),
        "realized_mdd_pct": round(_num(report.get("Max Realized Drawdown Percentage")), 4),
        "trades": int(report.get("Num Trades", 0)),
        "win_rate_pct": round(_num(report.get("Win Rate Percentage")), 4),
        "fees": round(_num(report.get("Total Fees")), 4),
        "funding": round(_num(report.get("Total Funding")), 4),
    }


def _passes(summary: dict[str, Any], criteria: Criteria) -> bool:
    return (
        _num(summary["return_pct"]) >= criteria.min_return_pct
        and _num(summary["net_pf"]) >= criteria.min_pf
        and _num(summary["mdd_pct"]) <= criteria.max_mdd_pct
        and int(summary["trades"]) >= criteria.min_trades
    )


def _gate(name: str, passed: bool, observed: Any, threshold: Any, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _decision(gates: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [gate for gate in gates if not gate["pass"]]
    return {
        "status": "PROMOTE_TO_SHADOW" if not failed else "HOLD_AUTOMATED_PAPER",
        "ready_for_shadow": not failed,
        "ready_for_paper": False,
        "failed_gates": [gate["name"] for gate in failed],
        "rationale": (
            "All validation gates passed; candidate may enter shadow logging only."
            if not failed
            else "Candidate is not eligible for automated paper trading or shadow promotion until failed gates pass."
        ),
    }


def _score(summary: dict[str, Any]) -> float:
    if int(summary["trades"]) <= 0:
        return -1e9
    net_pf = max(0.01, _num(summary["net_pf"]))
    return _num(summary["return_pct"]) + 10.0 * math.log(net_pf) - 0.25 * _num(summary["mdd_pct"])


def _with_execution_assumptions(
    params: dict[str, Any],
    entry_delay_bars: int,
    slippage_bps: float,
    funding_rate_per_8h: float,
    intrabar_policy: str,
) -> dict[str, Any]:
    out = dict(params)
    out["entry_delay_bars"] = entry_delay_bars
    out["slippage_bps"] = slippage_bps
    out["funding_rate_per_8h"] = funding_rate_per_8h
    out["intrabar_policy"] = intrabar_policy
    return out


def _date_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        end_ts = pd.to_datetime(end, utc=True)
        if len(str(end).strip()) == 10 and str(end).count("-") == 2:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        out = out[out["timestamp"] <= end_ts]
    return out.reset_index(drop=True)


def _run_one(
    df: pd.DataFrame,
    params: dict[str, Any],
    initial: float,
    risk_pct: float,
    fee_bps: float,
) -> tuple[dict[str, Any], np.ndarray]:
    report, equity = simulate(
        df,
        params,
        initial_balance=initial,
        risk_pct=risk_pct,
        fee_bps=fee_bps,
        no_gate=False,
        diag=False,
    )
    return report, equity


def _equity_returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    prev = equity[:-1]
    curr = equity[1:]
    return np.divide(curr - prev, prev, out=np.zeros_like(curr), where=prev != 0)


def _moments(x: np.ndarray) -> tuple[float, float]:
    if x.size < 3:
        return 0.0, 3.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return 0.0, 3.0
    z = (x - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return skew, kurt


def _deflated_sharpe(equity: np.ndarray, n_trials: int) -> dict[str, float]:
    returns = _equity_returns(equity)
    n = int(returns.size)
    if n < 3:
        return {"observations": n, "annualized_sharpe": 0.0, "dsr": 0.0, "benchmark_sharpe": 0.0}
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return {"observations": n, "annualized_sharpe": 0.0, "dsr": 0.0, "benchmark_sharpe": 0.0}

    sr = mean / std
    annualized = sr * math.sqrt(24.0 * 365.0)
    skew, kurt = _moments(returns)
    trials = max(2, int(n_trials))
    normal = NormalDist()
    gamma = 0.5772156649
    sr_std = 1.0 / math.sqrt(max(1, n - 1))
    benchmark = sr_std * (
        (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trials)
        + gamma * normal.inv_cdf(1.0 - 1.0 / (math.e * trials))
    )
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark) * math.sqrt(n - 1) / denom
    dsr = normal.cdf(z)
    return {
        "observations": n,
        "annualized_sharpe": round(annualized, 6),
        "dsr": round(float(dsr), 6),
        "benchmark_sharpe": round(float(benchmark), 6),
        "skew": round(skew, 6),
        "kurtosis": round(kurt, 6),
    }


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(-np.min(dd) * 100.0)


def _monte_carlo(equity: np.ndarray, runs: int, seed: int, initial: float) -> dict[str, Any]:
    returns = _equity_returns(equity)
    if returns.size == 0 or runs <= 0:
        return {"runs": 0}
    rng = np.random.default_rng(seed)
    final_returns = []
    mdds = []
    for _ in range(runs):
        sample = rng.choice(returns, size=returns.size, replace=True)
        path = initial * np.cumprod(1.0 + sample)
        path = np.concatenate([[initial], path])
        final_returns.append((path[-1] - initial) / initial * 100.0)
        mdds.append(_max_drawdown_from_equity(path))
    return {
        "runs": runs,
        "return_pct_p05": round(float(np.percentile(final_returns, 5)), 4),
        "return_pct_p50": round(float(np.percentile(final_returns, 50)), 4),
        "return_pct_p95": round(float(np.percentile(final_returns, 95)), 4),
        "mdd_pct_p50": round(float(np.percentile(mdds, 50)), 4),
        "mdd_pct_p95": round(float(np.percentile(mdds, 95)), 4),
        "prob_return_positive": round(float(np.mean(np.array(final_returns) > 0.0)), 4),
    }


def _grid(name: str, max_grid: int) -> list[dict[str, Any]]:
    grid = dense_grid() if name == "dense" else coarse_grid()
    return grid[:max_grid] if max_grid > 0 else grid


def _rolling_folds(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_bars: int,
    embargo_bars: int,
    min_rows: int,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    first = df["timestamp"].min().floor("h")
    last = df["timestamp"].max().ceil("h")
    folds: list[dict[str, Any]] = []
    fold_start = first
    while True:
        train_start = fold_start
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last + pd.Timedelta(hours=1):
            break

        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        test = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)]
        if purge_bars > 0 and len(train) > purge_bars:
            train = train.iloc[:-purge_bars]
        if embargo_bars > 0 and len(test) > embargo_bars:
            test = test.iloc[embargo_bars:]
        if len(train) >= min_rows and len(test) >= min_rows:
            folds.append(
                {
                    "fold": len(folds) + 1,
                    "train_start": train["timestamp"].min(),
                    "train_end": train["timestamp"].max(),
                    "test_start": test["timestamp"].min(),
                    "test_end": test["timestamp"].max(),
                    "train_df": train.reset_index(drop=True),
                    "test_df": test.reset_index(drop=True),
                }
            )
        fold_start = fold_start + pd.DateOffset(months=step_months)
    return folds


def _select_best(
    df: pd.DataFrame,
    grid: list[dict[str, Any]],
    initial: float,
    risk_pct: float,
    fee_bps: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for idx, params in enumerate(grid):
        report, equity = _run_one(df, params, initial, risk_pct, fee_bps)
        summary = _report_summary(report)
        score = _score(summary)
        if best is None or score > best["score"]:
            best = {
                "grid_index": idx,
                "score": score,
                "params": params,
                "report": report,
                "summary": summary,
                "equity": equity,
            }
    if best is None:
        raise RuntimeError("No strategy candidates were evaluated.")
    return best


def _walk_forward(
    folds: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    criteria: Criteria,
    initial: float,
    risk_pct: float,
    fee_bps: float,
) -> dict[str, Any]:
    rows = []
    pass_count = 0
    for fold in folds:
        selected = _select_best(fold["train_df"], grid, initial, risk_pct, fee_bps)
        oos_report, oos_equity = _run_one(fold["test_df"], selected["params"], initial, risk_pct, fee_bps)
        oos_summary = _report_summary(oos_report)
        oos_pass = _passes(oos_summary, criteria)
        pass_count += int(oos_pass)
        rows.append(
            {
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "selected_grid_index": selected["grid_index"],
                "is_summary": selected["summary"],
                "oos_summary": oos_summary,
                "oos_pass": oos_pass,
                "oos_dsr": _deflated_sharpe(oos_equity, len(grid)),
            }
        )
    pass_ratio = pass_count / len(rows) if rows else 0.0
    return {"folds": rows, "pass_ratio": round(pass_ratio, 4), "positive_folds": pass_count, "num_folds": len(rows)}


def _block_reports(
    df: pd.DataFrame,
    grid: list[dict[str, Any]],
    splits: int,
    initial: float,
    risk_pct: float,
    fee_bps: float,
) -> np.ndarray:
    df_reset = df.reset_index(drop=True)
    block_edges = np.linspace(0, len(df_reset), splits + 1, dtype=int)
    blocks = [df_reset.iloc[block_edges[i] : block_edges[i + 1]] for i in range(splits)]
    scores = np.full((len(grid), splits), -1e9, dtype=np.float64)
    for p_idx, params in enumerate(grid):
        for b_idx, block in enumerate(blocks):
            if len(block) < 10:
                continue
            report, _ = _run_one(block.reset_index(drop=True), params, initial, risk_pct, fee_bps)
            scores[p_idx, b_idx] = _score(_report_summary(report))
    return scores


def _cscv_pbo(
    df: pd.DataFrame,
    grid: list[dict[str, Any]],
    splits: int,
    initial: float,
    risk_pct: float,
    fee_bps: float,
) -> dict[str, Any]:
    if splits < 4 or splits % 2 != 0:
        return {"enabled": False, "reason": "splits must be an even integer >= 4"}
    scores = _block_reports(df, grid, splits, initial, risk_pct, fee_bps)
    block_indices = range(splits)
    combos = list(itertools.combinations(block_indices, splits // 2))
    lambdas = []
    ranks = []
    for train_blocks in combos:
        train_set = set(train_blocks)
        test_blocks = [idx for idx in block_indices if idx not in train_set]
        train_scores = np.nanmean(scores[:, list(train_blocks)], axis=1)
        test_scores = np.nanmean(scores[:, test_blocks], axis=1)
        if not np.isfinite(train_scores).any() or not np.isfinite(test_scores).any():
            continue
        winner = int(np.nanargmax(train_scores))
        order = np.argsort(np.argsort(test_scores))
        pct_rank = float(order[winner] / max(1, len(grid) - 1))
        pct_rank = min(1.0 - 1e-9, max(1e-9, pct_rank))
        lambdas.append(math.log(pct_rank / (1.0 - pct_rank)))
        ranks.append(pct_rank)
    if not lambdas:
        return {"enabled": False, "reason": "no finite CSCV samples"}
    lambdas_arr = np.array(lambdas, dtype=np.float64)
    ranks_arr = np.array(ranks, dtype=np.float64)
    return {
        "enabled": True,
        "splits": splits,
        "samples": int(len(lambdas)),
        "pbo": round(float(np.mean(lambdas_arr <= 0.0)), 4),
        "median_test_percentile": round(float(np.median(ranks_arr)), 4),
        "lambda_median": round(float(np.median(lambdas_arr)), 6),
    }


def _cost_stress(
    df: pd.DataFrame,
    params: dict[str, Any],
    criteria: Criteria,
    initial: float,
    risk_pct: float,
    base_fee_bps: float,
    base_slippage_bps: float,
    funding_values: list[float],
    fee_multipliers: list[float],
    slippage_multipliers: list[float],
) -> dict[str, Any]:
    rows = []
    pass_count = 0
    for fee_mult in fee_multipliers:
        for slip_mult in slippage_multipliers:
            for funding in funding_values:
                stressed = dict(params)
                stressed["slippage_bps"] = base_slippage_bps * slip_mult
                stressed["funding_rate_per_8h"] = funding
                report, _ = _run_one(df, stressed, initial, risk_pct, base_fee_bps * fee_mult)
                summary = _report_summary(report)
                passed = _passes(summary, criteria)
                pass_count += int(passed)
                rows.append(
                    {
                        "fee_multiplier": fee_mult,
                        "slippage_multiplier": slip_mult,
                        "funding_rate_per_8h": funding,
                        "summary": summary,
                        "pass": passed,
                    }
                )
    return {"rows": rows, "pass_ratio": round(pass_count / len(rows), 4) if rows else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v2 EMA strategy with WFO, CSCV/PBO, DSR, MC, and cost stress.")
    parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-08-14")
    parser.add_argument("--grid", choices=["coarse", "dense"], default="dense")
    parser.add_argument("--max-grid", type=int, default=64)
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=2.0, help="Percent of equity risked per trade.")
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0001)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--intrabar-policy", choices=["conservative", "optimistic"], default="conservative")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-fold-rows", type=int, default=300)
    parser.add_argument("--cscv-splits", type=int, default=6)
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-wfo-pass-ratio", type=float, default=0.6)
    parser.add_argument("--max-pbo", type=float, default=0.5)
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--min-mc-prob-positive", type=float, default=0.6)
    parser.add_argument("--min-mc-return-p05", type=float, default=0.0)
    parser.add_argument("--min-stress-pass-ratio", type=float, default=0.5)
    parser.add_argument("--out", default="wfa_optimized_params_output/v2_validation_report.json")
    parser.add_argument("--fail-on-hold", action="store_true", help="Return exit code 2 when the decision is HOLD_AUTOMATED_PAPER.")
    args = parser.parse_args()

    criteria = Criteria(
        min_return_pct=args.min_return,
        min_pf=args.min_pf,
        max_mdd_pct=args.max_mdd,
        min_trades=args.min_trades,
        min_wfo_pass_ratio=args.min_wfo_pass_ratio,
        max_pbo=args.max_pbo,
        min_dsr=args.min_dsr,
        min_mc_prob_positive=args.min_mc_prob_positive,
        min_mc_return_p05=args.min_mc_return_p05,
        min_stress_pass_ratio=args.min_stress_pass_ratio,
    )
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    df = _date_filter(load_csv(str(csv_path)), args.start, args.end)
    grid = [
        _with_execution_assumptions(
            params,
            args.entry_delay_bars,
            args.slippage_bps,
            args.funding_rate_per_8h,
            args.intrabar_policy,
        )
        for params in _grid(args.grid, args.max_grid)
    ]
    if df.empty:
        raise SystemExit("No data in requested validation window.")
    if not grid:
        raise SystemExit("No parameter candidates.")

    risk_pct = args.risk / 100.0
    folds = _rolling_folds(
        df,
        args.train_months,
        args.test_months,
        args.step_months,
        args.purge_bars,
        args.embargo_bars,
        args.min_fold_rows,
    )
    full_selected = _select_best(df, grid, args.initial, risk_pct, args.fee_bps)
    full_summary = full_selected["summary"]
    full_dsr = _deflated_sharpe(full_selected["equity"], len(grid))
    wfo = _walk_forward(folds, grid, criteria, args.initial, risk_pct, args.fee_bps)
    pbo = _cscv_pbo(df, grid, args.cscv_splits, args.initial, risk_pct, args.fee_bps)
    mc = _monte_carlo(full_selected["equity"], args.mc_runs, args.seed, args.initial)
    stress = _cost_stress(
        df,
        full_selected["params"],
        criteria,
        args.initial,
        risk_pct,
        args.fee_bps,
        args.slippage_bps,
        [0.0, args.funding_rate_per_8h, args.funding_rate_per_8h * 2.0],
        [1.0, 1.5, 2.0],
        [1.0, 2.0],
    )

    pbo_value = _num(pbo.get("pbo"), default=1.0) if pbo.get("enabled") else 1.0
    gates = [
        _gate("full_return_pct", full_summary["return_pct"] >= criteria.min_return_pct, full_summary["return_pct"], f">= {criteria.min_return_pct}", "Expanded-period net return."),
        _gate("full_net_pf", full_summary["net_pf"] >= criteria.min_pf, full_summary["net_pf"], f">= {criteria.min_pf}", "Net PF after fees/slippage/funding."),
        _gate("full_mdd_pct", full_summary["mdd_pct"] <= criteria.max_mdd_pct, full_summary["mdd_pct"], f"<= {criteria.max_mdd_pct}", "Intrabar mark-to-market MDD."),
        _gate("full_trades", full_summary["trades"] >= criteria.min_trades, full_summary["trades"], f">= {criteria.min_trades}", "Minimum trade count."),
        _gate("wfo_pass_ratio", wfo["pass_ratio"] >= criteria.min_wfo_pass_ratio, wfo["pass_ratio"], f">= {criteria.min_wfo_pass_ratio}", "Rolling WFO pass ratio."),
        _gate("pbo", pbo.get("enabled") and pbo_value <= criteria.max_pbo, pbo.get("pbo", "disabled"), f"<= {criteria.max_pbo}", "CSCV/PBO."),
        _gate("dsr", _num(full_dsr["dsr"]) >= criteria.min_dsr, full_dsr["dsr"], f">= {criteria.min_dsr}", "Deflated Sharpe Ratio."),
        _gate("mc_prob_return_positive", _num(mc.get("prob_return_positive"), default=0.0) >= criteria.min_mc_prob_positive, mc.get("prob_return_positive", 0.0), f">= {criteria.min_mc_prob_positive}", "Monte Carlo bootstrap positive-return probability."),
        _gate("mc_return_p05", _num(mc.get("return_pct_p05"), default=-float("inf")) >= criteria.min_mc_return_p05, mc.get("return_pct_p05", -float("inf")), f">= {criteria.min_mc_return_p05}", "Monte Carlo 5th percentile return."),
        _gate("cost_stress_pass_ratio", stress["pass_ratio"] >= criteria.min_stress_pass_ratio, stress["pass_ratio"], f">= {criteria.min_stress_pass_ratio}", "Fee/slippage/funding stress matrix pass ratio."),
    ]
    decision = _decision(gates)
    strict_ready = decision["ready_for_shadow"]

    report = {
        "schema_version": 1,
        "data": {
            "csv": str(csv_path),
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max(),
            "rows": len(df),
        },
        "assumptions": {
            "market": "Binance USD-M perpetual",
            "entry_delay_bars": args.entry_delay_bars,
            "intrabar_policy": args.intrabar_policy,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
            "funding_rate_per_8h": args.funding_rate_per_8h,
            "purge_bars": args.purge_bars,
            "embargo_bars": args.embargo_bars,
        },
        "criteria": criteria.__dict__,
        "grid": {"name": args.grid, "evaluated": len(grid)},
        "full_sample": {
            "selected_grid_index": full_selected["grid_index"],
            "summary": full_summary,
            "dsr": full_dsr,
            "params": full_selected["params"],
        },
        "walk_forward": wfo,
        "cscv_pbo": pbo,
        "monte_carlo": mc,
        "cost_stress": stress,
        "gates": gates,
        "decision": decision,
        "strict_ready_for_shadow": strict_ready,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    print(f"wrote {out_path}")
    print(
        "summary "
        f"full_return={full_summary['return_pct']} "
        f"full_pf={full_summary['net_pf']} "
        f"full_mdd={full_summary['mdd_pct']} "
        f"wfo_pass_ratio={wfo['pass_ratio']} "
        f"pbo={pbo.get('pbo', 'n/a')} "
        f"dsr={full_dsr['dsr']} "
        f"stress_pass_ratio={stress['pass_ratio']} "
        f"strict_ready={strict_ready}"
    )
    return 2 if args.fail_on_hold and not strict_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
