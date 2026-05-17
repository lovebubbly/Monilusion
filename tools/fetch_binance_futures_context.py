from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://fapi.binance.com"
FUNDING_RATE_PATH = "/fapi/v1/fundingRate"
PREMIUM_INDEX_KLINES_PATH = "/fapi/v1/premiumIndexKlines"
OPEN_INTEREST_HIST_PATH = "/futures/data/openInterestHist"
INTERVAL_MS = {"1h": 60 * 60_000, "8h": 8 * 60 * 60_000}


def _parse_time_ms(value: str) -> int:
    return int(pd.to_datetime(value, utc=True).timestamp() * 1000)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def _request_json(path: str, params: dict[str, Any], *, timeout: float) -> Any:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(
        f"{BASE_URL}{path}?{query}",
        headers={"User-Agent": "MonilusionFuturesContext/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_paginated(
    *,
    path: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout: float,
    sleep_seconds: float,
    next_from_row,
    extra_params: dict[str, Any] | None = None,
    max_pages: int = 0,
) -> list[Any]:
    rows: list[Any] = []
    cur = start_ms
    pages = 0
    while cur <= end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": limit}
        if extra_params:
            params.update(extra_params)
        batch = _request_json(path, params, timeout=timeout)
        if isinstance(batch, dict):
            raise RuntimeError(f"Binance returned an error object for {path}: {batch}")
        if not batch:
            break
        rows.extend(batch)
        pages += 1
        next_ms = int(next_from_row(batch[-1]))
        if next_ms <= cur:
            next_ms = cur + 1
        cur = next_ms
        if len(batch) < limit:
            break
        if max_pages and pages >= max_pages:
            break
        time.sleep(sleep_seconds)
    return rows


def _funding_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
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
    return pd.DataFrame(parsed).drop_duplicates(subset=["funding_time"]).sort_values("funding_time")


def _premium_rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    parsed = []
    for row in rows:
        if len(row) < 7:
            continue
        open_time = int(row[0])
        parsed.append(
            {
                "open_time": open_time,
                "open_time_iso": _iso(open_time),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "close_time": int(row[6]),
                "close_time_iso": _iso(int(row[6])),
            }
        )
    return pd.DataFrame(parsed).drop_duplicates(subset=["open_time"]).sort_values("open_time")


def _open_interest_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed = []
    for row in rows:
        ts = int(row["timestamp"])
        parsed.append(
            {
                "timestamp": ts,
                "timestamp_iso": _iso(ts),
                "symbol": row.get("symbol"),
                "sum_open_interest": float(row.get("sumOpenInterest", 0.0)),
                "sum_open_interest_value": float(row.get("sumOpenInterestValue", 0.0)),
            }
        )
    return pd.DataFrame(parsed).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    time_col = "funding_time" if "funding_time" in frame.columns else "open_time" if "open_time" in frame.columns else "timestamp"
    first_ms = int(frame[time_col].min()) if not frame.empty else None
    last_ms = int(frame[time_col].max()) if not frame.empty else None
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "first_timestamp": _iso(first_ms),
        "last_timestamp": _iso(last_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Binance USD-M futures context data for strategy validation.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("wfa_optimized_params_output/futures_context"))
    parser.add_argument("--datasets", default="funding,premium", help="Comma list: funding,premium,open_interest")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages per dataset for smoke tests.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    start_ms = _parse_time_ms(args.start)
    end_ms = _parse_time_ms(args.end)
    if end_ms < start_ms:
        raise SystemExit("--end must be after --start")

    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    stamp = f"{pd.to_datetime(start_ms, unit='ms', utc=True).strftime('%Y%m%d')}_{pd.to_datetime(end_ms, unit='ms', utc=True).strftime('%Y%m%d')}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "binance_usd_m_futures_context_fetch",
        "base_url": BASE_URL,
        "symbol": args.symbol,
        "requested_start": _iso(start_ms),
        "requested_end": _iso(end_ms),
        "datasets_requested": datasets,
        "dry_run": bool(args.dry_run),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "outputs": {},
        "warnings": [],
    }

    if args.dry_run:
        for dataset in datasets:
            if dataset == "funding":
                manifest["outputs"][dataset] = {"endpoint": FUNDING_RATE_PATH, "planned": True}
            elif dataset == "premium":
                manifest["outputs"][dataset] = {"endpoint": PREMIUM_INDEX_KLINES_PATH, "planned": True}
            elif dataset == "open_interest":
                manifest["outputs"][dataset] = {
                    "endpoint": OPEN_INTEREST_HIST_PATH,
                    "planned": True,
                    "note": "Binance USD-M open interest history is often window-limited; use as auxiliary evidence unless full coverage is available.",
                }
            else:
                manifest["warnings"].append(f"unknown dataset ignored: {dataset}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if "funding" in datasets:
        rows = _fetch_paginated(
            path=FUNDING_RATE_PATH,
            symbol=args.symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=min(args.limit, 1000),
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            next_from_row=lambda row: int(row["fundingTime"]) + 1,
            max_pages=args.max_pages,
        )
        frame = _funding_rows_to_frame(rows)
        manifest["outputs"]["funding"] = {
            "endpoint": FUNDING_RATE_PATH,
            **_write_csv(frame, out_dir / f"{args.symbol}_funding_rate_8h_{stamp}.csv"),
        }

    if "premium" in datasets:
        rows = _fetch_paginated(
            path=PREMIUM_INDEX_KLINES_PATH,
            symbol=args.symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=min(args.limit, 1500),
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            extra_params={"interval": "1h"},
            next_from_row=lambda row: int(row[0]) + INTERVAL_MS["1h"],
            max_pages=args.max_pages,
        )
        frame = _premium_rows_to_frame(rows)
        manifest["outputs"]["premium"] = {
            "endpoint": PREMIUM_INDEX_KLINES_PATH,
            "interval": "1h",
            **_write_csv(frame, out_dir / f"{args.symbol}_premium_index_1h_{stamp}.csv"),
        }

    if "open_interest" in datasets:
        rows = _fetch_paginated(
            path=OPEN_INTEREST_HIST_PATH,
            symbol=args.symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=min(args.limit, 500),
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            extra_params={"period": "1h"},
            next_from_row=lambda row: int(row["timestamp"]) + INTERVAL_MS["1h"],
            max_pages=args.max_pages,
        )
        frame = _open_interest_rows_to_frame(rows)
        manifest["outputs"]["open_interest"] = {
            "endpoint": OPEN_INTEREST_HIST_PATH,
            "period": "1h",
            "note": "Treat as auxiliary unless audit confirms broad historical coverage.",
            **_write_csv(frame, out_dir / f"{args.symbol}_open_interest_1h_{stamp}.csv"),
        }

    manifest_path = out_dir / f"{args.symbol}_futures_context_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
