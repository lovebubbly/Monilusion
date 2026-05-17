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


USD_M_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _parse_time_ms(value: str | None) -> int | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True)
    return int(ts.timestamp() * 1000)


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


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in df.columns}
    time_col = cols.get("timestamp") or cols.get("open time") or cols.get("date")
    if time_col is None:
        raise SystemExit(f"Cannot find timestamp/Open time column in {path}")
    rename = {
        time_col: "timestamp",
        cols.get("open", "open"): "open",
        cols.get("high", "high"): "high",
        cols.get("low", "low"): "low",
        cols.get("close", "close"): "close",
        cols.get("volume", "volume"): "volume",
    }
    missing = [name for name in ["open", "high", "low", "close", "volume"] if name not in cols]
    if missing:
        raise SystemExit(f"Missing OHLCV columns in {path}: {missing}")
    out = df.rename(columns=rename)[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    parsed = _parse_ts(out["timestamp"])
    out["_parsed_time"] = parsed
    out = out.dropna(subset=["_parsed_time"])
    out["timestamp"] = (out["_parsed_time"].astype("int64") // 1_000_000).astype("int64")
    out = out.drop(columns=["_parsed_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return (
        out.dropna()
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def klines_to_frame(rows: list[list[Any]], *, now_ms: int) -> pd.DataFrame:
    parsed = []
    for row in rows:
        if len(row) < 7:
            continue
        close_time = int(row[6])
        if close_time >= now_ms:
            continue
        parsed.append(
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    if not parsed:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(parsed)


def fetch_klines(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout: float,
) -> list[list[Any]]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    req = urllib.request.Request(f"{USD_M_KLINES_URL}?{params}", headers={"User-Agent": "MonilusionShadow/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_range(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout: float,
    sleep_seconds: float,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    cur = start_ms
    step_ms = INTERVAL_MS[interval]
    while cur < end_ms:
        batch = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=cur,
            end_ms=end_ms,
            limit=limit,
            timeout=timeout,
        )
        if not batch:
            break
        rows.extend(batch)
        next_ms = int(batch[-1][0]) + step_ms
        if next_ms <= cur:
            next_ms = cur + step_ms
        cur = next_ms
        if len(batch) < limit:
            break
        time.sleep(sleep_seconds)
    return rows


def merge_ohlcv(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = fetched.copy()
    elif fetched.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, fetched], ignore_index=True)
    return (
        combined.dropna()
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def write_ohlcv_csv(path: Path, frame: pd.DataFrame, *, backup: bool) -> Path | None:
    backup_path = None
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(path, backup_path)
    frame.to_csv(path, index=False)
    return backup_path


def update_ohlcv(
    *,
    csv_path: Path,
    symbol: str,
    interval: str,
    start_ms: int | None,
    end_ms: int | None,
    overlap_bars: int,
    limit: int,
    timeout: float,
    sleep_seconds: float,
    write: bool,
    backup: bool,
    mock_rows: list[list[Any]] | None = None,
) -> dict[str, Any]:
    if interval not in INTERVAL_MS:
        raise SystemExit(f"Unsupported interval {interval!r}; supported={sorted(INTERVAL_MS)}")
    now_ms = _now_ms()
    effective_end_ms = min(end_ms or now_ms, now_ms)
    existing = load_ohlcv_csv(csv_path)
    last_existing_ms = int(existing["timestamp"].max()) if not existing.empty else None
    if start_ms is None:
        if last_existing_ms is None:
            raise SystemExit("--start is required when the CSV has no existing rows.")
        start_ms = max(0, last_existing_ms - overlap_bars * INTERVAL_MS[interval])
    raw_rows = mock_rows if mock_rows is not None else fetch_range(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=effective_end_ms,
        limit=limit,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
    )
    fetched = klines_to_frame(raw_rows, now_ms=now_ms)
    merged = merge_ohlcv(existing, fetched)
    backup_path = write_ohlcv_csv(csv_path, merged, backup=backup) if write else None
    first_new_ms = int(fetched["timestamp"].min()) if not fetched.empty else None
    last_new_ms = int(fetched["timestamp"].max()) if not fetched.empty else None
    return {
        "schema_version": 1,
        "mode": "binance_usd_m_futures_ohlcv_update",
        "endpoint": USD_M_KLINES_URL,
        "symbol": symbol,
        "interval": interval,
        "csv_path": str(csv_path),
        "write": write,
        "backup_path": str(backup_path) if backup_path else None,
        "rows_before": int(len(existing)),
        "raw_rows_fetched": int(len(raw_rows)),
        "closed_rows_fetched": int(len(fetched)),
        "rows_after": int(len(merged)),
        "net_new_rows": int(len(merged) - len(existing)),
        "overlap_refreshed_rows": int(max(0, len(fetched) - max(0, len(merged) - len(existing)))),
        "last_existing_timestamp": pd.to_datetime(last_existing_ms, unit="ms", utc=True).isoformat()
        if last_existing_ms is not None
        else None,
        "fetch_start_timestamp": pd.to_datetime(start_ms, unit="ms", utc=True).isoformat(),
        "fetch_end_timestamp": pd.to_datetime(effective_end_ms, unit="ms", utc=True).isoformat(),
        "first_fetched_closed_timestamp": pd.to_datetime(first_new_ms, unit="ms", utc=True).isoformat()
        if first_new_ms is not None
        else None,
        "last_fetched_closed_timestamp": pd.to_datetime(last_new_ms, unit="ms", utc=True).isoformat()
        if last_new_ms is not None
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update local OHLCV CSV from Binance USD-M futures klines.")
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
    parser.add_argument("--start", default=None, help="UTC start timestamp. Required for an empty CSV.")
    parser.add_argument("--end", default=None, help="UTC end timestamp. Defaults to now.")
    parser.add_argument("--overlap-bars", type=int, default=6)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--write", action="store_true", help="Write merged rows back to --csv. Omit for dry-run.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file when writing.")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--mock-json", type=Path, default=None, help="Use a local Binance kline JSON payload instead of network.")
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    mock_rows = None
    if args.mock_json is not None:
        mock_path = args.mock_json if args.mock_json.is_absolute() else Path.cwd() / args.mock_json
        mock_rows = json.loads(mock_path.read_text(encoding="utf-8"))
    summary = update_ohlcv(
        csv_path=csv_path,
        symbol=args.symbol,
        interval=args.interval,
        start_ms=_parse_time_ms(args.start),
        end_ms=_parse_time_ms(args.end),
        overlap_bars=args.overlap_bars,
        limit=args.limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        write=args.write,
        backup=not args.no_backup,
        mock_rows=mock_rows,
    )
    if args.summary_out is not None:
        out = args.summary_out if args.summary_out.is_absolute() else Path.cwd() / args.summary_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
