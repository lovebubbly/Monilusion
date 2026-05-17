from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative


def _num(value: Any) -> float:
    if value == "inf":
        return float("inf")
    return float(value)


def _parse_ranks(value: str | None, max_rank: int) -> set[int]:
    if value:
        out: set[int] = set()
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                out.update(range(int(start_s), int(end_s) + 1))
            else:
                out.add(int(part))
        return out
    return set(range(1, max_rank + 1))


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": metrics.get("total_net_pnl_percentage"),
        "net_pf": metrics.get("net_profit_factor", metrics.get("profit_factor")),
        "gross_pf": metrics.get("gross_profit_factor"),
        "mdd_pct": metrics.get("max_drawdown_percentage"),
        "trades": metrics.get("num_trades"),
        "win_rate_pct": metrics.get("win_rate_percentage"),
        "total_funding": metrics.get("total_funding"),
        "funding_model": metrics.get("funding_model"),
    }


def _delta(actual: dict[str, Any], constant: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "return_pct": ("total_net_pnl_percentage", 4),
        "net_pf": ("net_profit_factor", 4),
        "mdd_pct": ("max_drawdown_percentage", 4),
        "total_funding": ("total_funding", 4),
    }
    out = {}
    for name, (key, digits) in keys.items():
        out[name] = round(_num(actual.get(key)) - _num(constant.get(key)), digits)
    return out


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Actual Funding Impact",
        "",
        f"- Source CUDA JSON: `{report['cuda_json']}`",
        f"- Funding CSV: `{report['funding_csv']}`",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Rows: `{report['period']['rows']}`",
        f"- Diagnostic only: `{report['diagnostic_only']}`",
        "",
        "## Candidates",
    ]
    for row in report["candidates"]:
        d = row["delta_actual_minus_constant"]
        a = row["actual_funding_summary"]
        c = row["constant_funding_summary"]
        lines.append(
            f"- rank {row['rank']}: actual_return=`{a['return_pct']}%`, constant_return=`{c['return_pct']}%`, "
            f"delta_return=`{d['return_pct']}%`, actual_pf=`{a['net_pf']}`, actual_mdd=`{a['mdd_pct']}%`, "
            f"delta_funding=`{d['total_funding']}`"
        )
    lines.extend(
        [
            "",
            "## Note",
            "- This report reprices CPU reference results with actual funding events. It is not a promotion artifact until CUDA search/validation is run with the same actual funding model and diff-tested.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare constant funding vs actual Binance funding events for saved CUDA candidates.")
    parser.add_argument("--cuda-json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--funding-csv", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--ranks", default=None)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    cuda_json = args.cuda_json if args.cuda_json.is_absolute() else root / args.cuda_json
    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    funding_csv = args.funding_csv if args.funding_csv.is_absolute() else root / args.funding_csv
    obj = json.loads(cuda_json.read_text(encoding="utf-8"))
    df = _load_ohlcv(csv_path, args.start, args.end)
    funding_cumulative = load_funding_cumulative(df, funding_csv)
    ranks = _parse_ranks(args.ranks, args.max_rank)
    rows = [row for row in obj.get("results", []) if int(row.get("rank", 0)) in ranks]
    rows.sort(key=lambda row: int(row["rank"]))

    commission = float(obj.get("commission_rate", 0.0005))
    slippage = float(obj.get("slippage_rate", 0.0002))
    entry_delay = int(obj.get("entry_delay_bars", 1))
    constant_funding = float(obj.get("funding_rate_per_8h", 0.0))
    candidates = []
    for row in rows:
        params = row["parameters"]
        constant_metrics = cpu_reference_backtest(
            df,
            params,
            initial_balance=args.initial,
            commission_rate=commission,
            slippage_rate=slippage,
            entry_delay_bars=entry_delay,
            funding_rate_per_8h=constant_funding,
        )
        actual_metrics = cpu_reference_backtest(
            df,
            params,
            initial_balance=args.initial,
            commission_rate=commission,
            slippage_rate=slippage,
            entry_delay_bars=entry_delay,
            funding_rate_per_8h=constant_funding,
            funding_cumulative=funding_cumulative,
        )
        candidates.append(
            {
                "rank": int(row["rank"]),
                "param_id": row["performance"]["param_id"],
                "constant_funding_summary": _summary(constant_metrics),
                "actual_funding_summary": _summary(actual_metrics),
                "delta_actual_minus_constant": _delta(actual_metrics, constant_metrics),
            }
        )

    report = {
        "schema_version": 1,
        "mode": "actual_funding_impact",
        "cuda_json": str(cuda_json),
        "csv": str(csv_path),
        "funding_csv": str(funding_csv),
        "period": {
            "start": pd.to_datetime(args.start, utc=True).isoformat(),
            "end": pd.to_datetime(args.end, utc=True).isoformat(),
            "rows": int(len(df)),
        },
        "assumptions": {
            "commission_rate": commission,
            "slippage_rate": slippage,
            "entry_delay_bars": entry_delay,
            "constant_funding_rate_per_8h": constant_funding,
            "actual_funding_timing": "funding events after entry bar through exit bar are charged; event at the entry bar is excluded",
        },
        "diagnostic_only": True,
        "candidates": candidates,
    }
    if args.out_json:
        out = args.out_json if args.out_json.is_absolute() else root / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out = args.out_md if args.out_md.is_absolute() else root / args.out_md
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
