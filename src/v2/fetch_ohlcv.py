# -*- coding: utf-8 -*-
# src/v2/fetch_ohlcv.py
from __future__ import annotations
import argparse, time, math
from datetime import datetime, timezone
from dateutil import parser as dtparser
import requests
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"
INTERVALS = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
             "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
             "1d":"1d","3d":"3d","1w":"1w","1M":"1M"}

def to_millis(dt_str: str) -> int:
    # "2020-01-01", "2020-01-01T00:00:00Z" 등 파싱
    dt = dtparser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit=1000):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser(description="Download OHLCV from Binance (spot)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", choices=list(INTERVALS.keys()))
    ap.add_argument("--start", required=True, help="e.g. 2019-01-01")
    ap.add_argument("--end", default=None, help="e.g. 2025-08-14 (UTC). Omit = now")
    ap.add_argument("--out", required=True, help="output csv path, e.g. data/BTCUSDT_1h.csv")
    args = ap.parse_args()

    start_ms = to_millis(args.start)
    end_ms = to_millis(args.end) if args.end else int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    cur = start_ms
    while cur < end_ms:
        data = fetch_klines(args.symbol, args.interval, cur, end_ms, limit=1000)
        if not data:
            break
        for k in data:
            # kline schema: [openTime, open, high, low, close, volume, closeTime, ...]
            rows.append({
                "timestamp": int(k[0]),  # ms
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        next_ms = int(data[-1][6]) + 1  # last closeTime + 1ms
        if next_ms <= cur:
            # 안전장치
            cur += 1_000
        else:
            cur = next_ms
        time.sleep(0.1)  # API 과다 호출 방지

    if not rows:
        raise SystemExit("No data fetched. Check symbol/interval/date range.")

    df = pd.DataFrame(rows)
    # timestamp(ms) → UTC ISO로 바꿔도 되지만 backtest_v2가 ms도 처리함.
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
