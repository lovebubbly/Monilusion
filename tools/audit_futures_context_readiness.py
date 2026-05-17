from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_ts(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        unit = "ms"
        finite = numeric.dropna()
        if not finite.empty:
            max_abs = finite.abs().max()
            if max_abs >= 1e14:
                unit = "us"
            elif max_abs < 1e11:
                unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _load_ohlcv_window(path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in df.columns}
    time_col = cols.get("timestamp") or cols.get("open time") or cols.get("date")
    if time_col is None:
        raise SystemExit(f"Cannot find timestamp/Open time column in {path}")
    out = pd.DataFrame({"timestamp": _parse_ts(df[time_col])}).dropna()
    if start:
        out = out[out["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        end_ts = pd.to_datetime(end, utc=True)
        if len(str(end).strip()) == 10 and str(end).count("-") == 2:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        out = out[out["timestamp"] <= end_ts]
    return out.sort_values("timestamp").reset_index(drop=True)


def _latest_matching(context_dir: Path, pattern: str) -> Path | None:
    paths = sorted(context_dir.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return paths[0] if paths else None


def _coverage_summary(
    *,
    path: Path | None,
    time_col: str,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    expected_interval_hours: float,
    max_gap_hours: float,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "ready": False,
            "path": None,
            "rows": 0,
            "coverage_ratio": 0.0,
            "large_gaps": [],
        }
    frame = pd.read_csv(path)
    if time_col not in frame.columns:
        return {
            "status": "bad_schema",
            "ready": False,
            "path": str(path),
            "rows": int(len(frame)),
            "coverage_ratio": 0.0,
            "large_gaps": [],
            "error": f"missing column {time_col}",
        }
    ts = _parse_ts(frame[time_col]).dropna().sort_values().drop_duplicates()
    ts = ts[(ts >= expected_start) & (ts <= expected_end)]
    expected_count = max(1, int(((expected_end - expected_start).total_seconds() / 3600.0) / expected_interval_hours) + 1)
    diffs = ts.diff().dropna().dt.total_seconds() / 3600.0
    large_gaps = [
        {
            "after": ts.iloc[i - 1].isoformat(),
            "before": ts.iloc[i].isoformat(),
            "gap_hours": round(float((ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 3600.0), 4),
        }
        for i in range(1, len(ts))
        if (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 3600.0 > max_gap_hours
    ][:20]
    first = ts.min() if not ts.empty else None
    last = ts.max() if not ts.empty else None
    coverage_ratio = min(1.0, float(len(ts)) / expected_count)
    start_ok = first is not None and first <= expected_start + pd.Timedelta(hours=max_gap_hours)
    end_ok = last is not None and last >= expected_end - pd.Timedelta(hours=max_gap_hours)
    ready = bool(coverage_ratio >= 0.95 and start_ok and end_ok and not large_gaps)
    return {
        "status": "ready" if ready else "incomplete",
        "ready": ready,
        "path": str(path),
        "rows": int(len(ts)),
        "expected_rows_approx": int(expected_count),
        "coverage_ratio": round(coverage_ratio, 4),
        "first_timestamp": first.isoformat() if first is not None else None,
        "last_timestamp": last.isoformat() if last is not None else None,
        "start_ok": bool(start_ok),
        "end_ok": bool(end_ok),
        "max_observed_gap_hours": round(float(diffs.max()), 4) if not diffs.empty else None,
        "large_gaps": large_gaps,
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Futures Context Readiness Audit",
        "",
        f"- Symbol: `{report['symbol']}`",
        f"- OHLCV rows: `{report['ohlcv']['rows']}`",
        f"- OHLCV start: `{report['ohlcv']['start']}`",
        f"- OHLCV end: `{report['ohlcv']['end']}`",
        f"- Overall decision: `{report['decision']}`",
        "",
        "## Datasets",
    ]
    for name, item in report["datasets"].items():
        lines.extend(
            [
                f"- `{name}`: status=`{item['status']}`, ready=`{item['ready']}`, rows=`{item['rows']}`, coverage=`{item.get('coverage_ratio')}`",
                f"  - path: `{item.get('path')}`",
                f"  - first: `{item.get('first_timestamp')}`, last: `{item.get('last_timestamp')}`",
                f"  - max_gap_hours: `{item.get('max_observed_gap_hours')}`",
            ]
        )
        if item.get("large_gaps"):
            lines.append(f"  - large_gaps_sample: `{len(item['large_gaps'])}`")
    lines.extend(["", "## Recommendations"])
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local Binance futures context coverage for strategy validation.")
    parser.add_argument("--ohlcv-csv", type=Path, default=Path("wfa_optimized_params_output/live_shadow_phase14_data/BTCUSDT_1h.csv"))
    parser.add_argument("--context-dir", type=Path, default=Path("wfa_optimized_params_output/futures_context"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    ohlcv_path = args.ohlcv_csv if args.ohlcv_csv.is_absolute() else Path.cwd() / args.ohlcv_csv
    context_dir = args.context_dir if args.context_dir.is_absolute() else Path.cwd() / args.context_dir
    ohlcv = _load_ohlcv_window(ohlcv_path, args.start, args.end)
    if ohlcv.empty:
        raise SystemExit("OHLCV window is empty")

    expected_start = ohlcv["timestamp"].min()
    expected_end = ohlcv["timestamp"].max()
    funding_path = _latest_matching(context_dir, f"{args.symbol}_funding_rate_8h_*.csv")
    premium_path = _latest_matching(context_dir, f"{args.symbol}_premium_index_1h_*.csv")
    oi_path = _latest_matching(context_dir, f"{args.symbol}_open_interest_1h_*.csv")
    datasets = {
        "funding_rate_8h": _coverage_summary(
            path=funding_path,
            time_col="funding_time",
            expected_start=expected_start,
            expected_end=expected_end,
            expected_interval_hours=8.0,
            max_gap_hours=16.0,
        ),
        "premium_index_1h": _coverage_summary(
            path=premium_path,
            time_col="open_time",
            expected_start=expected_start,
            expected_end=expected_end,
            expected_interval_hours=1.0,
            max_gap_hours=3.0,
        ),
        "open_interest_1h": _coverage_summary(
            path=oi_path,
            time_col="timestamp",
            expected_start=expected_start,
            expected_end=expected_end,
            expected_interval_hours=1.0,
            max_gap_hours=3.0,
        ),
    }
    recommendations = []
    if not datasets["funding_rate_8h"]["ready"]:
        recommendations.append("Fetch full funding history before replacing the current constant funding-rate model.")
    if not datasets["premium_index_1h"]["ready"]:
        recommendations.append("Fetch premium index klines before testing funding/premium crowding filters.")
    if not datasets["open_interest_1h"]["ready"]:
        recommendations.append("Treat OI as live-shadow auxiliary data unless broad historical coverage is available.")
    decision = "READY_FOR_ACTUAL_FUNDING_RESEARCH" if datasets["funding_rate_8h"]["ready"] else "FETCH_REQUIRED"
    report = {
        "schema_version": 1,
        "mode": "futures_context_readiness_audit",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": args.symbol,
        "ohlcv": {
            "path": str(ohlcv_path),
            "rows": int(len(ohlcv)),
            "start": expected_start.isoformat(),
            "end": expected_end.isoformat(),
        },
        "context_dir": str(context_dir),
        "datasets": datasets,
        "decision": decision,
        "recommendations": recommendations,
    }
    if args.out_json:
        out = args.out_json if args.out_json.is_absolute() else Path.cwd() / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out = args.out_md if args.out_md.is_absolute() else Path.cwd() / args.out_md
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
