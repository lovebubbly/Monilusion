from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative
from validate_v2_strategy import _deflated_sharpe, _equity_returns, _max_drawdown_from_equity, _moments


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _resolve(path: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _profit_factor(values: np.ndarray) -> float | str:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0.0:
        return "inf" if gains > 0.0 else 0.0
    return round(gains / losses, 6)


def _required_annualized_sharpe(dsr: dict[str, Any], target: float) -> float | None:
    n = int(dsr.get("observations", 0))
    if n < 3:
        return None
    benchmark = _num(dsr.get("benchmark_sharpe"))
    skew = _num(dsr.get("skew"))
    kurt = _num(dsr.get("kurtosis"), 3.0)
    normal = NormalDist()

    def dsr_for_sr(sr: float) -> float:
        denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
        z = (sr - benchmark) * math.sqrt(n - 1) / denom
        return normal.cdf(z)

    lo, hi = -1.0, 1.0
    while dsr_for_sr(hi) < target and hi < 16.0:
        hi *= 2.0
    if dsr_for_sr(hi) < target:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if dsr_for_sr(mid) >= target:
            hi = mid
        else:
            lo = mid
    return round(hi * math.sqrt(24.0 * 365.0), 6)


def _return_distribution(returns: np.ndarray) -> dict[str, Any]:
    if returns.size == 0:
        return {"observations": 0}
    nonzero = returns[np.abs(returns) > 1e-12]
    out: dict[str, Any] = {
        "observations": int(returns.size),
        "nonzero_observations": int(nonzero.size),
        "zero_return_ratio": round(1.0 - (nonzero.size / returns.size), 6),
        "mean": round(float(np.mean(returns)), 10),
        "std": round(float(np.std(returns, ddof=1)), 10) if returns.size > 1 else 0.0,
        "min": round(float(np.min(returns)), 10),
        "p01": round(float(np.percentile(returns, 1)), 10),
        "p05": round(float(np.percentile(returns, 5)), 10),
        "median": round(float(np.median(returns)), 10),
        "p95": round(float(np.percentile(returns, 95)), 10),
        "p99": round(float(np.percentile(returns, 99)), 10),
        "max": round(float(np.max(returns)), 10),
    }
    if nonzero.size:
        skew, kurt = _moments(nonzero)
        out.update(
            {
                "nonzero_mean": round(float(np.mean(nonzero)), 10),
                "nonzero_std": round(float(np.std(nonzero, ddof=1)), 10) if nonzero.size > 1 else 0.0,
                "nonzero_skew": round(skew, 6),
                "nonzero_kurtosis": round(kurt, 6),
                "nonzero_p05": round(float(np.percentile(nonzero, 5)), 10),
                "nonzero_median": round(float(np.median(nonzero)), 10),
                "nonzero_p95": round(float(np.percentile(nonzero, 95)), 10),
            }
        )
    return out


def _trade_distribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0}
    ordered = sorted(trades, key=lambda row: str(row.get("exit_time", row.get("entry_signal_time", ""))))
    pnl = np.array([_num(row.get("net_pnl")) for row in ordered], dtype=np.float64)
    skew, kurt = _moments(pnl)
    max_loss_run = 0
    current_loss_run = 0
    worst_loss_run_pnl = 0.0
    current_loss_run_pnl = 0.0
    for value in pnl:
        if value < 0.0:
            current_loss_run += 1
            current_loss_run_pnl += float(value)
            max_loss_run = max(max_loss_run, current_loss_run)
            worst_loss_run_pnl = min(worst_loss_run_pnl, current_loss_run_pnl)
        else:
            current_loss_run = 0
            current_loss_run_pnl = 0.0
    return {
        "trades": int(pnl.size),
        "win_rate_pct": round(float(np.mean(pnl > 0.0)) * 100.0, 6),
        "profit_factor": _profit_factor(pnl),
        "mean_net_pnl": round(float(np.mean(pnl)), 6),
        "median_net_pnl": round(float(np.median(pnl)), 6),
        "std_net_pnl": round(float(np.std(pnl, ddof=1)), 6) if pnl.size > 1 else 0.0,
        "skew": round(skew, 6),
        "kurtosis": round(kurt, 6),
        "p05_net_pnl": round(float(np.percentile(pnl, 5)), 6),
        "p95_net_pnl": round(float(np.percentile(pnl, 95)), 6),
        "max_consecutive_losses": int(max_loss_run),
        "worst_consecutive_loss_pnl": round(float(worst_loss_run_pnl), 6),
    }


def _combine_equity(equity_curves: list[np.ndarray], initial: float) -> np.ndarray:
    combined = [float(initial)]
    current = float(initial)
    for curve in equity_curves:
        if curve.size < 2:
            continue
        returns = _equity_returns(curve)
        for ret in returns:
            current *= 1.0 + float(ret)
            combined.append(current)
    return np.array(combined, dtype=np.float64)


def _combine_weighted_equity(equity_curves: list[np.ndarray], weights: np.ndarray, initial: float) -> np.ndarray:
    min_len = min((curve.size for curve in equity_curves), default=0)
    if min_len == 0:
        return np.array([initial], dtype=np.float64)
    combined = np.zeros(min_len, dtype=np.float64)
    for curve, weight in zip(equity_curves, weights):
        combined += curve[:min_len] * float(weight)
    return combined


def _official_summary(equity: np.ndarray, trades: list[dict[str, Any]], official: dict[str, Any] | None) -> dict[str, Any]:
    if official:
        return {
            "final_balance": round(_num(official.get("final_balance"), float(equity[-1]) if equity.size else 10_000.0), 6),
            "return_pct": round(_num(official.get("return_pct", official.get("total_net_pnl_percentage"))), 6),
            "mdd_pct": round(_num(official.get("mdd_pct", official.get("max_drawdown_percentage"))), 6),
            "trades": int(_num(official.get("trades", official.get("num_trades")), len(trades))),
        }
    final_balance = float(equity[-1]) if equity.size else 10_000.0
    initial_balance = float(equity[0]) if equity.size else 10_000.0
    return {
        "final_balance": round(final_balance, 6),
        "return_pct": round((final_balance - initial_balance) / initial_balance * 100.0, 6) if initial_balance else 0.0,
        "mdd_pct": round(_max_drawdown_from_equity(equity), 6),
        "trades": len(trades),
    }


def _shape(
    label: str,
    equity: np.ndarray,
    trades: list[dict[str, Any]],
    n_trials: int,
    *,
    official_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    returns = _equity_returns(equity)
    dsr = _deflated_sharpe(equity, n_trials)
    return {
        "label": label,
        "summary": _official_summary(equity, trades, official_summary),
        "dsr": dsr,
        "required_annualized_sharpe": {
            "dsr_0_80": _required_annualized_sharpe(dsr, 0.80),
            "dsr_0_95": _required_annualized_sharpe(dsr, 0.95),
        },
        "return_distribution": _return_distribution(returns),
        "trade_distribution": _trade_distribution(trades),
    }


def _select_rank(cuda_obj: dict[str, Any], rank: int) -> dict[str, Any]:
    for row in cuda_obj.get("results", []):
        if int(row.get("rank", -1)) == rank:
            return row
    raise SystemExit(f"No rank {rank} in {cuda_obj.get('output_file', '<cuda json>')}")


def _analyze_batch(path: Path, rank: int) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    csv = _resolve(obj["data"]["csv"])
    funding_csv = _resolve(obj.get("assumptions", {}).get("funding_rate_csv"))
    source_cuda = _resolve(obj["source_cuda_json"], base=path.parent)
    if source_cuda is None or not source_cuda.exists():
        source_cuda = _resolve(obj["source_cuda_json"])
    cuda_obj = json.loads(source_cuda.read_text(encoding="utf-8"))
    row = _select_rank(cuda_obj, rank)
    df = _load_ohlcv(csv, obj["data"]["period_start"], obj["data"]["period_end"])
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    metrics, equity, trades = cpu_reference_backtest(
        df,
        row["parameters"],
        initial_balance=10_000.0,
        commission_rate=_num(obj.get("assumptions", {}).get("commission_rate")),
        slippage_rate=_num(obj.get("assumptions", {}).get("slippage_rate")),
        entry_delay_bars=int(obj.get("assumptions", {}).get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(obj.get("assumptions", {}).get("funding_rate_per_8h")),
        include_equity=True,
        include_trades=True,
        funding_cumulative=funding_cumulative,
    )
    trials = int(obj.get("candidate_universe", {}).get("dsr_trials", cuda_obj.get("total_param_combinations", 2)))
    out = _shape(f"{path.stem}:rank{rank}", equity, trades, trials, official_summary=metrics)
    out["source"] = {"type": "batch", "path": str(path), "cuda_json": str(source_cuda), "rank": rank}
    out["saved_summary"] = metrics
    out["pbo"] = obj.get("topk_cscv_pbo")
    return out


def _eval_params(
    csv: Path,
    start: str,
    end: str,
    params: dict[str, Any],
    assumptions: dict[str, Any],
    initial: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    df = _load_ohlcv(csv, start, end)
    funding_csv = _resolve(assumptions.get("funding_rate_csv"))
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    _, equity, trades = cpu_reference_backtest(
        df,
        params,
        initial_balance=initial,
        commission_rate=_num(assumptions.get("commission_rate")),
        slippage_rate=_num(assumptions.get("slippage_rate")),
        entry_delay_bars=int(assumptions.get("entry_delay_bars", 1)),
        funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h")),
        include_equity=True,
        include_trades=True,
        funding_cumulative=funding_cumulative,
    )
    return equity, trades


def _analyze_wfo(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    csv = _resolve(obj["csv"])
    initial = 10_000.0
    assumptions = obj.get("assumptions", {})
    fold_equities: list[np.ndarray] = []
    all_trades: list[dict[str, Any]] = []
    for fold in obj.get("folds", []):
        selections = []
        if fold.get("selected_ensemble"):
            selections = list(fold["selected_ensemble"])
        elif fold.get("selected_parameters"):
            selections = [{"parameters": fold["selected_parameters"]}]
        if not selections:
            fold_equities.append(np.array([initial], dtype=np.float64))
            continue
        equities = []
        weights = np.full(len(selections), 1.0 / len(selections), dtype=np.float64)
        for idx, selection in enumerate(selections):
            equity, trades = _eval_params(
                csv,
                fold["test_start"],
                fold["test_end"],
                selection["parameters"],
                assumptions,
                initial,
            )
            equities.append(equity)
            for trade in trades:
                scaled = dict(trade)
                for key in ("gross_pnl", "net_pnl", "commission", "slippage_cost", "funding"):
                    if key in scaled:
                        scaled[key] = _num(scaled[key]) * float(weights[idx])
                all_trades.append(scaled)
        fold_equities.append(_combine_weighted_equity(equities, weights, initial))
    combined = _combine_equity(fold_equities, initial)
    trials = int(obj.get("summary", {}).get("dsr_trials", 2))
    out = _shape(path.stem, combined, all_trades, trials, official_summary=obj.get("aggregate"))
    out["source"] = {"type": "wfo", "path": str(path)}
    out["saved_aggregate"] = obj.get("aggregate")
    out["saved_dsr"] = obj.get("dsr")
    return out


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        "label",
        "return",
        "mdd",
        "trades",
        "ann_sharpe",
        "dsr",
        "req95",
        "kurt",
        "zero_ratio",
        "trade_pf",
        "max_loss_run",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        dsr = row["dsr"]
        ret = row["return_distribution"]
        trade = row["trade_distribution"]
        values = [
            row["label"],
            row["summary"]["return_pct"],
            row["summary"]["mdd_pct"],
            row["summary"]["trades"],
            dsr.get("annualized_sharpe"),
            dsr.get("dsr"),
            row["required_annualized_sharpe"].get("dsr_0_95"),
            dsr.get("kurtosis"),
            ret.get("zero_return_ratio"),
            trade.get("profit_factor"),
            trade.get("max_consecutive_losses"),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return lines


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DSR Return-Shape Diagnostics",
        "",
        "This diagnostic mirrors the current strict DSR formula. Active-only or trade-level stats are explanatory only and do not replace promotion gates.",
        "",
    ]
    lines.extend(_markdown_table(rows))
    lines.extend(["", "## Interpretation Notes", ""])
    lines.append("- `req95` is the annualized Sharpe needed for DSR 0.95 if the current skew/kurtosis and trial count stay unchanged.")
    lines.append("- `zero_ratio` is the share of hourly equity returns that are exactly flat; high values usually mean sparse exposure.")
    lines.append("- Promotion still requires the original strict gates; this file is for research triage.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose DSR blockers by equity-return and trade-return shape.")
    parser.add_argument("--batch-json", action="append", type=Path, default=[])
    parser.add_argument("--wfo-json", action="append", type=Path, default=[])
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in args.batch_json:
        rows.append(_analyze_batch(_resolve(path), args.rank))
    for path in args.wfo_json:
        rows.append(_analyze_wfo(_resolve(path)))
    result = {"schema_version": 1, "rank": args.rank, "diagnostics": rows}
    text = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    if args.out_json:
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
    if args.out_md:
        out_md = _resolve(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, rows)
    if args.out_json or args.out_md:
        print(json.dumps({"rows": len(rows), "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
