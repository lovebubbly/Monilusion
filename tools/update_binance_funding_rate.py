from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


USD_M_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_INTERVAL_MS = 8 * 60 * 60_000


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _parse_time_ms(value: str | None) -> int | None:
    if not value:
        return None
    return int(pd.to_datetime(value, utc=True).timestamp() * 1000)


def _parse_ts(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(values) * 0.9)):
        finite = numeric.dropna()
        unit = "ms"
        if not finite.empty:
            max_abs = finite.abs().max()
            if max_abs >= 1e14:
                unit = "us"
            elif max_abs < 1e11:
                unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def load_funding_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["funding_time", "funding_time_iso", "symbol", "funding_rate", "mark_price"])
    frame = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in frame.columns}
    time_col = cols.get("funding_time") or cols.get("timestamp") or cols.get("time")
    rate_col = cols.get("funding_rate") or cols.get("fundingrate")
    if time_col is None or rate_col is None:
        raise SystemExit(f"Funding CSV must contain funding_time/timestamp and funding_rate columns: {path}")
    symbol_col = cols.get("symbol")
    mark_col = cols.get("mark_price") or cols.get("markprice")
    parsed_time = _parse_ts(frame[time_col])
    out = pd.DataFrame(
        {
            "_parsed_time": parsed_time,
            "symbol": frame[symbol_col] if symbol_col is not None else None,
            "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce"),
            "mark_price": pd.to_numeric(frame[mark_col], errors="coerce") if mark_col is not None else None,
        }
    )
    out = out.dropna(subset=["_parsed_time", "funding_rate"])
    out["funding_time"] = (out["_parsed_time"].astype("int64") // 1_000_000).astype("int64")
    out = out.drop(columns=["_parsed_time"])
    out["funding_time"] = out["funding_time"].astype("int64")
    out["funding_time_iso"] = pd.to_datetime(out["funding_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    out["funding_time_iso"] = out["funding_time_iso"].str.replace(r"\.000000\+0000$", "+00:00", regex=True)
    out["funding_time_iso"] = out["funding_time_iso"].str.replace("+0000", "+00:00", regex=False)
    return (
        out[["funding_time", "funding_time_iso", "symbol", "funding_rate", "mark_price"]]
        .drop_duplicates(subset=["funding_time"], keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def fetch_funding_rates(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout: float,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur = start_ms
    while cur <= end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "startTime": cur,
                "endTime": end_ms,
                "limit": min(limit, 1000),
            }
        )
        req = urllib.request.Request(f"{USD_M_FUNDING_URL}?{params}", headers={"User-Agent": "MonilusionFunding/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if isinstance(batch, dict):
            raise RuntimeError(f"Binance returned an error object for fundingRate: {batch}")
        if not batch:
            break
        rows.extend(batch)
        next_ms = int(batch[-1]["fundingTime"]) + 1
        if next_ms <= cur:
            next_ms = cur + 1
        cur = next_ms
        if len(batch) < min(limit, 1000):
            break
        time.sleep(sleep_seconds)
    return rows


def funding_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed = []
    for row in rows:
        funding_time = int(row["fundingTime"])
        parsed.append(
            {
                "funding_time": funding_time,
                "funding_time_iso": _iso(funding_time),
                "symbol": row.get("symbol"),
                "funding_rate": float(row["fundingRate"]),
                "mark_price": float(row["markPrice"]) if row.get("markPrice") not in {None, ""} else None,
            }
        )
    if not parsed:
        return pd.DataFrame(columns=["funding_time", "funding_time_iso", "symbol", "funding_rate", "mark_price"])
    return pd.DataFrame(parsed).drop_duplicates(subset=["funding_time"], keep="last").sort_values("funding_time")


def merge_funding(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = fetched.copy()
    elif fetched.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        return combined
    if "symbol" not in combined.columns:
        combined["symbol"] = None
    if "mark_price" not in combined.columns:
        combined["mark_price"] = None
    combined["funding_time"] = pd.to_numeric(combined["funding_time"], errors="coerce")
    combined["funding_rate"] = pd.to_numeric(combined["funding_rate"], errors="coerce")
    combined["mark_price"] = pd.to_numeric(combined["mark_price"], errors="coerce")
    combined = combined.dropna(subset=["funding_time", "funding_rate"])
    combined["funding_time"] = combined["funding_time"].astype("int64")
    combined["funding_time_iso"] = pd.to_datetime(combined["funding_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    combined["funding_time_iso"] = combined["funding_time_iso"].str.replace(r"\.000000\+0000$", "+00:00", regex=True)
    combined["funding_time_iso"] = combined["funding_time_iso"].str.replace("+0000", "+00:00", regex=False)
    return (
        combined[["funding_time", "funding_time_iso", "symbol", "funding_rate", "mark_price"]]
        .drop_duplicates(subset=["funding_time"], keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def write_funding_csv(path: Path, frame: pd.DataFrame, *, backup: bool) -> Path | None:
    backup_path = None
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(path, backup_path)
    frame.to_csv(path, index=False)
    return backup_path


def update_funding_rates(
    *,
    csv_path: Path,
    symbol: str,
    start_ms: int | None,
    end_ms: int | None,
    overlap_events: int,
    limit: int,
    timeout: float,
    sleep_seconds: float,
    write: bool,
    backup: bool,
    mock_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now_ms = _now_ms()
    effective_end_ms = min(end_ms or now_ms, now_ms)
    existing = load_funding_csv(csv_path)
    last_existing_ms = int(existing["funding_time"].max()) if not existing.empty else None
    if start_ms is None:
        if last_existing_ms is None:
            raise SystemExit("--start is required when the funding CSV has no existing rows.")
        start_ms = max(0, last_existing_ms - overlap_events * FUNDING_INTERVAL_MS)
    raw_rows = mock_rows if mock_rows is not None else fetch_funding_rates(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=effective_end_ms,
        limit=limit,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
    )
    fetched = funding_rows_to_frame(raw_rows)
    merged = merge_funding(existing, fetched)
    backup_path = write_funding_csv(csv_path, merged, backup=backup) if write else None
    first_new_ms = int(fetched["funding_time"].min()) if not fetched.empty else None
    last_new_ms = int(fetched["funding_time"].max()) if not fetched.empty else None
    return {
        "schema_version": 1,
        "mode": "binance_usd_m_funding_rate_update",
        "endpoint": USD_M_FUNDING_URL,
        "symbol": symbol,
        "csv_path": str(csv_path),
        "write": write,
        "backup_path": str(backup_path) if backup_path else None,
        "rows_before": int(len(existing)),
        "raw_rows_fetched": int(len(raw_rows)),
        "rows_fetched": int(len(fetched)),
        "rows_after": int(len(merged)),
        "net_new_rows": int(len(merged) - len(existing)),
        "overlap_refreshed_rows": int(max(0, len(fetched) - max(0, len(merged) - len(existing)))),
        "last_existing_timestamp": _iso(last_existing_ms),
        "fetch_start_timestamp": _iso(start_ms),
        "fetch_end_timestamp": _iso(effective_end_ms),
        "first_fetched_timestamp": _iso(first_new_ms),
        "last_fetched_timestamp": _iso(last_new_ms),
        "last_merged_timestamp": _iso(int(merged["funding_time"].max())) if not merged.empty else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update local Binance USD-M funding-rate CSV in place.")
    parser.add_argument("--csv", type=Path, default=Path("wfa_optimized_params_output/futures_context/BTCUSDT_funding_rate_8h_20190101_20260516.csv"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--overlap-events", type=int, default=3)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--write", action="store_true", help="Write merged rows back to --csv. Omit for dry-run.")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--mock-json", type=Path, default=None)
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    mock_rows = None
    if args.mock_json is not None:
        mock_path = args.mock_json if args.mock_json.is_absolute() else Path.cwd() / args.mock_json
        mock_rows = json.loads(mock_path.read_text(encoding="utf-8"))
    summary = update_funding_rates(
        csv_path=csv_path,
        symbol=args.symbol,
        start_ms=_parse_time_ms(args.start),
        end_ms=_parse_time_ms(args.end),
        overlap_events=args.overlap_events,
        limit=args.limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        write=args.write,
        backup=not args.no_backup,
        mock_rows=mock_rows,
    )
    if args.summary_out is not None:
        summary_path = args.summary_out if args.summary_out.is_absolute() else Path.cwd() / args.summary_out
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
