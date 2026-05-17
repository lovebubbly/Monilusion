# -*- coding: utf-8 -*-
# src/v2/real_m1_v2.py
from __future__ import annotations
import os, time, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from .strategy_components import build_indicators, regime_gate, session_mask, long_entry_mask, short_entry_mask
from .exits import HybridExit

class BrokerStub:
    def __init__(self, fee_bps=2.0):
        self.fee_bps = fee_bps
        self.position = None  # (direction, entry_price, qty, entry_index)
        self.equity = 10_000.0

    def market_order(self, side: str, price: float, qty: float):
        # 실제 거래소 주문 API로 바꿔 끼우기
        # side: "BUY" or "SELL" (현물 기준)
        pass

def run_realtime(df_loader, poll_secs=2,
                 params=None, risk_pct=0.02, fee_bps=2.0):
    """
    df_loader(): callable -> 최신 OHLCV DataFrame 반환(분봉/시간봉)
    """
    broker = BrokerStub(fee_bps=fee_bps)
    last_ts = None
    exit_logic = HybridExit(sl_atr_mult=params.get("sl_k",2.6),
                            rr=params.get("rr",3.0),
                            atr_trail_mult=params.get("trail_k",1.0),
                            time_stop_bars=params.get("time_stop",48),
                            fee_bps=fee_bps)
    consec_losses = 0
    cb_max = params.get("cb_max_losses", 4)
    cb_cool = params.get("cb_cool_bars", 48)
    cool_until_ts = None

    while True:
        df = df_loader()
        if df is None or len(df) < 300:
            time.sleep(poll_secs); continue

        ts = df["timestamp"].iloc[-1]
        if last_ts is not None and ts == last_ts:
            time.sleep(poll_secs); continue
        last_ts = ts

        indis = build_indicators(df,
                                 ema_short=params.get("ema_short",14),
                                 ema_long=params.get("ema_long",100),
                                 rsi_n=params.get("rsi_n",21),
                                 adx_n=params.get("adx_n",14),
                                 atr_n=params.get("atr_n",21))
        close = df["close"].to_numpy(dtype=np.float64)
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)

        gate = regime_gate(indis, params.get("min_adx",15.0), params.get("min_atr_pct",0.002), close)
        sess = session_mask(df["timestamp"], params.get("allow_hours", []))
        tradable = gate & sess

        long_mask  = long_entry_mask(df, indis, params.get("long_rsi_low",52), params.get("long_rsi_high",60))
        short_mask = short_entry_mask(df, indis, params.get("short_rsi_low",40), params.get("short_rsi_high",48))
        long_entries  = tradable & long_mask
        short_entries = tradable & short_mask

        i = len(df) - 2  # 직전 바에서 체결 가정
        if cool_until_ts is not None and df["timestamp"].iloc[i] <= cool_until_ts:
            time.sleep(poll_secs); continue

        entry_price = close[i]
        if long_entries[i] or short_entries[i]:
            direction = 1 if long_entries[i] else -1
            atr_now = indis["atr"][i]
            if atr_now <= 0:
                time.sleep(poll_secs); continue
            risk_amount = broker.equity * risk_pct
            qty = max(risk_amount / (exit_logic.sl_k * atr_now), 0.0)
            # 이 부분에서 실거래 주문 전송하도록 교체
            # broker.market_order("BUY" if direction==1 else "SELL", entry_price, qty)

            # 단순 시뮬: 즉시 백테스트식 청산
            j_exit, exit_price = exit_logic.apply(i, direction, entry_price, indis["atr"], high, low, close)
            pnl = (exit_price - entry_price) * qty if direction==1 else (entry_price - exit_price) * qty
            broker.equity += pnl
            if pnl > 0: consec_losses = 0
            else:
                consec_losses += 1
                if consec_losses >= cb_max:
                    cool_until_ts = df["timestamp"].iloc[min(j_exit + cb_cool, len(df)-1)]
            # 로깅
            print(json.dumps({
                "ts": str(ts),
                "dir": "LONG" if direction==1 else "SHORT",
                "entry": float(entry_price),
                "exit": float(exit_price),
                "pnl": float(pnl),
                "equity": float(broker.equity)
            }, ensure_ascii=False))

        time.sleep(poll_secs)

def csv_loader_factory(path: str):
    def _load():
        df = pd.read_csv(path)
        ts = pd.to_datetime(np.where(df["timestamp"]>1e12, df["timestamp"], df["timestamp"]*1000), utc=True, unit="ms")
        df["timestamp"] = ts
        df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)
        return df
    return _load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--risk", type=float, default=2.0)
    args = ap.parse_args()
    params = {
        "ema_short":14, "ema_long":100, "rsi_n":21, "adx_n":14, "atr_n":21,
        "sl_k":2.6, "rr":3.0, "trail_k":1.0, "time_stop":48,
        "min_adx":15.0, "min_atr_pct":0.002,
        "long_rsi_low":52, "long_rsi_high":60,
        "short_rsi_low":40, "short_rsi_high":48,
        "allow_hours": list(range(0,24)),
        "cb_max_losses": 4, "cb_cool_bars": 48,
    }
    loader = csv_loader_factory(args.csv)
    run_realtime(loader, poll_secs=2, params=params, risk_pct=args.risk/100.0, fee_bps=2.0)

if __name__ == "__main__":
    main()
