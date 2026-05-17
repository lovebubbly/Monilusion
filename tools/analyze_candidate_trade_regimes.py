from __future__ import annotations

import argparse
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

from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _profit_factor(values: pd.Series) -> float | str:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0.0:
        return "inf" if gains > 0.0 else 0.0
    return round(gains / losses, 4)


def _summary(frame: pd.DataFrame, initial_balance: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "net_pnl_pct_initial": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_net_pnl": 0.0,
            "median_net_pnl": 0.0,
            "median_bars_held": 0.0,
            "avg_funding": 0.0,
        }
    pnl = frame["net_pnl"].astype(float)
    trades = int(len(frame))
    return {
        "trades": trades,
        "net_pnl": round(float(pnl.sum()), 4),
        "net_pnl_pct_initial": round(float(pnl.sum()) / initial_balance * 100.0, 4),
        "win_rate_pct": round(float((pnl > 0).mean()) * 100.0, 4),
        "profit_factor": _profit_factor(pnl),
        "avg_net_pnl": round(float(pnl.mean()), 4),
        "median_net_pnl": round(float(pnl.median()), 4),
        "median_bars_held": round(float(frame["bars_held"].median()), 4),
        "avg_funding": round(float(frame["funding"].astype(float).mean()), 4),
    }


def _group_summary(
    frame: pd.DataFrame,
    group_col: str,
    initial_balance: float,
    *,
    sort_by: str = "net_pnl",
    ascending: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if frame.empty or group_col not in frame.columns:
        return []
    rows = []
    for key, bucket in frame.groupby(group_col, observed=False, dropna=False):
        row = {"bucket": str(key)}
        row.update(_summary(bucket, initial_balance))
        rows.append(row)
    rows.sort(key=lambda row: _num(row.get(sort_by)), reverse=not ascending)
    return rows if limit is None else rows[:limit]


def _select_rank(obj: dict[str, Any], rank: int) -> dict[str, Any]:
    for row in obj.get("results", []):
        if int(row.get("rank", -1)) == rank:
            return row
    raise SystemExit(f"No rank {rank} found.")


def _fold_windows(
    df: pd.DataFrame,
    *,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_bars: int,
    embargo_bars: int,
    min_rows: int,
) -> list[dict[str, Any]]:
    first = df.index.min().floor("h")
    last = df.index.max().ceil("h")
    start = first
    out: list[dict[str, Any]] = []
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
                }
            )
        start = start + pd.DateOffset(months=step_months)
    return out


def _fold_summary(trades: pd.DataFrame, folds: list[dict[str, Any]], initial_balance: float) -> list[dict[str, Any]]:
    rows = []
    for fold in folds:
        mask = (
            (trades["entry_signal_time"] >= fold["test_start"])
            & (trades["entry_signal_time"] <= fold["test_end"])
        )
        bucket = trades.loc[mask]
        row = {
            "fold": int(fold["fold"]),
            "test_start": fold["test_start"].date().isoformat(),
            "test_end": fold["test_end"].date().isoformat(),
        }
        row.update(_summary(bucket, initial_balance))
        rows.append(row)
    rows.sort(key=lambda row: (_num(row["net_pnl"]), _num(row["profit_factor"]), _num(row["trades"])))
    return rows


def _add_regime_columns(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_signal_time"] = pd.to_datetime(out["entry_signal_time"])
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["exit_time"] = pd.to_datetime(out["exit_time"])
    out["entry_month"] = out["entry_signal_time"].dt.to_period("M").astype(str)
    out["entry_quarter"] = out["entry_signal_time"].dt.to_period("Q").astype(str)
    out["entry_weekday"] = out["entry_day_of_week"].map(
        {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
    )
    out["entry_hour_bucket"] = pd.cut(
        out["entry_hour_utc"].astype(float),
        bins=[-1, 6, 12, 21, 23],
        labels=["00-06", "07-12", "13-21", "22-23"],
    )
    out["entry_adx_bucket"] = pd.cut(
        out["entry_adx"].astype(float),
        bins=[-math.inf, 20, 25, 35, math.inf],
        labels=["<20", "20-25", "25-35", ">=35"],
    )
    out["entry_ema_spread_atr_bucket"] = pd.cut(
        out["entry_ema_spread_atr"].astype(float),
        bins=[-math.inf, 1, 2, 3, 5, math.inf],
        labels=["<1", "1-2", "2-3", "3-5", ">=5"],
    )
    out["entry_h1_slope_pct_bucket"] = pd.cut(
        out["entry_h1_slope_pct"].astype(float),
        bins=[-math.inf, -0.5, 0, 0.5, 1, 2, math.inf],
        labels=["<-0.5", "-0.5-0", "0-0.5", "0.5-1", "1-2", ">=2"],
    )
    out["entry_h4_slope_pct_bucket"] = pd.cut(
        out["entry_h4_slope_pct"].astype(float),
        bins=[-math.inf, -1, 0, 1, 2, 4, math.inf],
        labels=["<-1", "-1-0", "0-1", "1-2", "2-4", ">=4"],
    )
    return out


def _trim_trade(row: pd.Series) -> dict[str, Any]:
    keys = [
        "entry_signal_time",
        "exit_time",
        "side",
        "exit_reason",
        "net_pnl",
        "gross_pnl",
        "funding",
        "bars_held",
        "entry_adx",
        "entry_ema_spread_atr",
        "entry_h1_slope_pct",
        "entry_h4_slope_pct",
        "entry_hour_utc",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        value = row.get(key)
        if isinstance(value, pd.Timestamp):
            out[key] = value.isoformat()
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 10) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows[:limit]:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    candidate = analysis["candidate"]
    metrics = analysis["cpu_reference"]
    lines = [
        "# Candidate Trade Regime Analysis",
        "",
        f"- Source: `{analysis['source_cuda_json']}`",
        f"- Rank: {candidate['rank']}",
        f"- Param ID: `{candidate['param_id']}`",
        f"- Return: {metrics['total_net_pnl_percentage']}%, net PF: {metrics['net_profit_factor']}, MDD: {metrics['max_drawdown_percentage']}%, trades: {metrics['num_trades']}",
        "",
        "## Worst WFO Test Windows",
    ]
    lines.extend(
        _markdown_table(
            analysis["wfo_test_windows"][:8],
            ["fold", "test_start", "test_end", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor"],
        )
    )
    lines.extend(["", "## Worst Months"])
    lines.extend(
        _markdown_table(
            analysis["by_month_worst"],
            ["bucket", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor"],
        )
    )
    lines.extend(["", "## Exit Reasons"])
    lines.extend(
        _markdown_table(
            analysis["by_exit_reason"],
            ["bucket", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor", "median_bars_held"],
        )
    )
    lines.extend(["", "## Side"])
    lines.extend(
        _markdown_table(
            analysis["by_side"],
            ["bucket", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor", "median_bars_held"],
        )
    )
    lines.extend(["", "## ADX Buckets"])
    lines.extend(
        _markdown_table(
            analysis["by_adx_bucket"],
            ["bucket", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor"],
        )
    )
    lines.extend(["", "## H4 Slope Buckets"])
    lines.extend(
        _markdown_table(
            analysis["by_h4_slope_bucket"],
            ["bucket", "trades", "net_pnl", "net_pnl_pct_initial", "win_rate_pct", "profit_factor"],
        )
    )
    lines.extend(["", "## Largest Loss Trades"])
    lines.extend(
        _markdown_table(
            analysis["largest_loss_trades"],
            ["entry_signal_time", "exit_time", "exit_reason", "net_pnl", "bars_held", "entry_adx", "entry_h4_slope_pct"],
            limit=8,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    obj = json.loads(args.cuda_json.read_text(encoding="utf-8"))
    row = _select_rank(obj, args.rank)
    df = _load_ohlcv(args.csv, obj["period_start"], obj["period_end"])
    initial_balance = float(row["performance"].get("initial_balance", args.initial_balance))
    funding_csv = args.funding_csv
    if funding_csv is None and obj.get("funding_model") == "actual_funding_events" and obj.get("funding_rate_csv"):
        funding_csv = Path(obj["funding_rate_csv"])
    if funding_csv is not None and not funding_csv.is_absolute():
        funding_csv = ROOT / funding_csv
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    result, trades_raw = cpu_reference_backtest(
        df,
        row["parameters"],
        initial_balance=initial_balance,
        commission_rate=float(obj.get("commission_rate", args.commission_rate)),
        slippage_rate=float(obj.get("slippage_rate", args.slippage_rate)),
        entry_delay_bars=int(obj.get("entry_delay_bars", args.entry_delay_bars)),
        funding_rate_per_8h=float(obj.get("funding_rate_per_8h", args.funding_rate_per_8h)),
        funding_cumulative=funding_cumulative,
        include_trades=True,
    )
    trades = _add_regime_columns(pd.DataFrame(trades_raw))
    folds = _fold_windows(
        df,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
        min_rows=args.min_rows,
    )
    analysis = {
        "source_cuda_json": str(args.cuda_json),
        "csv": str(args.csv),
        "period_start": obj["period_start"],
        "period_end": obj["period_end"],
        "assumptions": {
            "commission_rate": float(obj.get("commission_rate", args.commission_rate)),
            "slippage_rate": float(obj.get("slippage_rate", args.slippage_rate)),
            "entry_delay_bars": int(obj.get("entry_delay_bars", args.entry_delay_bars)),
            "funding_rate_per_8h": float(obj.get("funding_rate_per_8h", args.funding_rate_per_8h)),
            "funding_model": "actual_funding_events" if funding_cumulative is not None else "constant_per_8h",
            "funding_rate_csv": str(funding_csv) if funding_csv is not None else None,
        },
        "candidate": {
            "rank": int(row["rank"]),
            "param_id": row["performance"]["param_id"],
            "parameters": row["parameters"],
            "saved_performance": row["performance"],
        },
        "cpu_reference": result,
        "trade_log_check": {
            "metric_num_trades": int(result["num_trades"]),
            "logged_trades": int(len(trades)),
            "matches": int(result["num_trades"]) == int(len(trades)),
        },
        "overall_trade_summary": _summary(trades, initial_balance),
        "wfo_test_windows": _fold_summary(trades, folds, initial_balance),
        "by_month_worst": _group_summary(trades, "entry_month", initial_balance, limit=12),
        "by_quarter_worst": _group_summary(trades, "entry_quarter", initial_balance, limit=12),
        "by_exit_reason": _group_summary(trades, "exit_reason", initial_balance, ascending=False),
        "by_side": _group_summary(trades, "side", initial_balance, ascending=False),
        "by_hour_bucket": _group_summary(trades, "entry_hour_bucket", initial_balance),
        "by_weekday": _group_summary(trades, "entry_weekday", initial_balance),
        "by_adx_bucket": _group_summary(trades, "entry_adx_bucket", initial_balance),
        "by_ema_spread_bucket": _group_summary(trades, "entry_ema_spread_atr_bucket", initial_balance),
        "by_h1_slope_bucket": _group_summary(trades, "entry_h1_slope_pct_bucket", initial_balance),
        "by_h4_slope_bucket": _group_summary(trades, "entry_h4_slope_pct_bucket", initial_balance),
        "largest_loss_trades": [_trim_trade(row) for _, row in trades.nsmallest(12, "net_pnl").iterrows()],
        "largest_win_trades": [_trim_trade(row) for _, row in trades.nlargest(12, "net_pnl").iterrows()],
    }
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Break down a saved CUDA candidate by trade regimes.")
    parser.add_argument("--cuda-json", required=True, type=Path)
    parser.add_argument("--csv", default=Path("data/BTCUSDT_1h.csv"), type=Path)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0)
    parser.add_argument("--funding-csv", type=Path, default=None)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=72)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-rows", type=int, default=720)
    args = parser.parse_args()

    analysis = analyze(args)
    text = json.dumps(analysis, ensure_ascii=False, indent=2, default=_json_default)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.out_md, analysis)
    if args.out_json or args.out_md:
        print(
            json.dumps(
                {
                    "rank": analysis["candidate"]["rank"],
                    "param_id": analysis["candidate"]["param_id"],
                    "return_pct": analysis["cpu_reference"]["total_net_pnl_percentage"],
                    "net_pf": analysis["cpu_reference"]["net_profit_factor"],
                    "mdd_pct": analysis["cpu_reference"]["max_drawdown_percentage"],
                    "trades": analysis["cpu_reference"]["num_trades"],
                    "trade_log_matches": analysis["trade_log_check"]["matches"],
                    "out_json": str(args.out_json) if args.out_json else None,
                    "out_md": str(args.out_md) if args.out_md else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
