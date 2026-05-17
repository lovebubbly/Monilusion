# -*- coding: utf-8 -*-
# src/v2/backtest_v2.py  (HTF + ADX rising + RSI cross + pullback 적용)
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from .metrics import summarize_report
from .strategy_components import (
    build_indicators, regime_gate, session_mask,
    long_entry_mask, short_entry_mask, dmi_adx
)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise SystemExit("CSV must contain 'timestamp' column.")
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        ts = pd.to_datetime(np.where(ts > 1e12, ts, ts * 1000), utc=True, unit="ms")
    else:
        ts = pd.to_datetime(ts, utc=True)
    df["timestamp"] = ts
    ren = {"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
    need = ["open","high","low","close","volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}")
    df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)
    return df

def parse_hours(spec: str | None) -> list[int]:
    if spec is None or spec.strip() == "" or spec.strip().lower() == "all":
        return list(range(0,24))
    s = spec.strip().lower()
    if "-" in s and "," not in s:
        a,b = s.split("-",1); a,b = int(a), int(b)
        return list(range(a,b+1)) if a<=b else list(range(a,24))+list(range(0,b+1))
    out = []
    for t in s.split(","):
        t=t.strip()
        if t: out.append(int(t))
    return sorted(set([h for h in out if 0<=h<=23]))

# ----------- 새 필터: HTF 컨텍스트 -----------
def build_htf_context(df: pd.DataFrame,
                      htf: str = "4h",
                      ema_short: int = 14,
                      ema_long: int = 100,
                      adx_n: int = 14,
                      min_adx: float = 10.0):
    """
    1H df를 받아 HTF(4h 등)로 리샘플 → HTF EMA/ADX 계산 → 1H 타임라인으로 asof 매핑
    return: (htf_long_ok, htf_short_ok) boolean arrays
    """
    # 1) htf 문자열 소문자화 (pandas 경고 제거)
    htf = (htf or "4h").lower()

    # 2) 4h 리샘플 후 HTF 인디케이터 계산
    idx = df.set_index("timestamp")
    # H1 timestamps are candle-open times. Label each HTF bar at the time it
    # becomes known, so lower timeframes cannot see an unfinished HTF candle.
    agg = idx.resample(htf, label="right", closed="left").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()

    if len(agg) < max(ema_short, ema_long, adx_n) + 5:
        ones = np.ones(len(df), dtype=bool)
        return ones, ones

    # EMA (HTF): pandas ewm
    ema_s_htf = agg["close"].ewm(span=ema_short, adjust=False).mean().to_numpy()
    ema_l_htf = agg["close"].ewm(span=ema_long,  adjust=False).mean().to_numpy()

    # ADX (HTF)
    plus_di_h, minus_di_h, adx_h = dmi_adx(
        agg["high"].to_numpy(),
        agg["low"].to_numpy(),
        agg["close"].to_numpy(),
        adx_n
    )

    # 3) 4h → 1h 타임라인으로 매핑 (backward asof)
    htf_ctx = pd.DataFrame({
        "timestamp": agg.index,
        "ema_s_htf": ema_s_htf,
        "ema_l_htf": ema_l_htf,
        "adx_htf":   adx_h
    }).dropna()

    mapped = pd.merge_asof(df[["timestamp"]], htf_ctx, on="timestamp", direction="backward")
    mapped = mapped.ffill()

    ema_s = mapped["ema_s_htf"].to_numpy()
    ema_l = mapped["ema_l_htf"].to_numpy()
    adx_h = mapped["adx_htf"].to_numpy()

    htf_long_ok  = (ema_s > ema_l) & (adx_h >= min_adx)
    htf_short_ok = (ema_s < ema_l) & (adx_h >= min_adx)
    return htf_long_ok.astype(bool), htf_short_ok.astype(bool)

# ----------- 새 필터: ADX 상승 / RSI 크로스 / 풀백 -----------
def adx_rising_mask(adx: np.ndarray, min_slope: float = 0.0) -> np.ndarray:
    d = adx - np.roll(adx, 1)
    d[0] = 0.0
    return d >= min_slope

def rsi_cross_masks(rsi: np.ndarray, long_low: float, short_high: float):
    prev = np.roll(rsi, 1); prev[0] = rsi[0]
    cross_up   = (rsi >= long_low)  & (prev < long_low)
    cross_down = (rsi <= short_high) & (prev > short_high)
    return cross_up, cross_down
def pullback_ok(close: np.ndarray, ema_s: np.ndarray, atr: np.ndarray, k: float, direction: int):
    """direction: +1 long, -1 short"""
    if k <= 0:
        return np.ones_like(close, dtype=bool)
    if direction == 1:
        # 너무 과매수(ema_s 위로 k*ATR 이상 이탈)면 제외
        return close <= (ema_s + k * atr)
    else:
        # 너무 과매도(ema_s 아래로 k*ATR 이상 이탈)면 제외
        return close >= (ema_s - k * atr)

def simulate(df: pd.DataFrame,
             params: dict,
             initial_balance: float = 10_000.0,
             risk_pct: float = 0.02,
             fee_bps: float = 2.0,
             no_gate: bool = False,
             diag: bool = False):
    open_ = df["open"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    entry_delay_bars = int(params.get("entry_delay_bars", 1))
    slippage_bps = float(params.get("slippage_bps", 0.0))
    slippage_rate = slippage_bps / 10000.0
    fee_rate = float(fee_bps) / 10000.0
    funding_rate_per_8h = float(params.get("funding_rate_per_8h", 0.0))


    indis = build_indicators(df,
                             ema_short=params.get("ema_short",14),
                             ema_long=params.get("ema_long",100),
                             rsi_n=params.get("rsi_n",21),
                             adx_n=params.get("adx_n",14),
                             atr_n=params.get("atr_n",21))

    # 게이트
    if no_gate:
        gate = np.ones(len(df), dtype=bool)
    else:
        gate = regime_gate(indis,
                           min_adx=params.get("min_adx",15.0),
                           min_atr_pct=params.get("min_atr_pct",0.002),
                           close=close)

    # 세션
    allow_hours = params.get("allow_hours", list(range(0,24)))
    sess = session_mask(df["timestamp"], allow_hours)
    tradable = gate & sess

    # 기본 엔트리(EMA/RSI/DI)
    base_long  = long_entry_mask(df, indis,
                                 rsi_low=params.get("long_rsi_low",52),
                                 rsi_high=params.get("long_rsi_high",60),
                                 use_cross=params.get("use_cross", True),
                                 use_rsi_band=params.get("use_rsi_band", True))
    base_short = short_entry_mask(df, indis,
                                  rsi_low=params.get("short_rsi_low",40),
                                  rsi_high=params.get("short_rsi_high",48),
                                  use_cross=params.get("use_cross", True),
                                  use_rsi_band=params.get("use_rsi_band", True))
    allow_shorts = params.get("allow_shorts", True)
    if not allow_shorts:
        short_entries = np.zeros(len(df), dtype=bool)

    # 옵션 필터들
    # (1) HTF 컨펌
    if params.get("use_htf", False):
        htf_long_ok, htf_short_ok = build_htf_context(
            df, params.get("htf","4H"),
            params.get("htf_ema_short",14),
            params.get("htf_ema_long",100),
            params.get("htf_adx_n",14),
            params.get("htf_min_adx",12.0)
        )
    else:
        htf_long_ok  = np.ones(len(df), dtype=bool)
        htf_short_ok = np.ones(len(df), dtype=bool)

    # (2) ADX 상승
    if params.get("require_adx_rising", False) or params.get("min_adx_slope",0.0) > 0.0:
        rising = adx_rising_mask(indis["adx"], params.get("min_adx_slope",0.0))
    else:
        rising = np.ones(len(df), dtype=bool)

    # (3) RSI 크로스
    if params.get("use_rsi_cross", False):
        cross_up, cross_down = rsi_cross_masks(indis["rsi"],
                                               params.get("long_rsi_low",52),
                                               params.get("short_rsi_high",48))
    else:
        cross_up  = np.ones(len(df), dtype=bool)
        cross_down= np.ones(len(df), dtype=bool)

    # (4) 풀백
    pull_long  = pullback_ok(close, indis["ema_s"], indis["atr"], params.get("entry_pullback_k",0.0), +1)
    pull_short = pullback_ok(close, indis["ema_s"], indis["atr"], params.get("entry_pullback_k",0.0), -1)

    # 최종 엔트리
    long_entries  = tradable & base_long  & htf_long_ok  & rising & cross_up  & pull_long

    allow_shorts = params.get("allow_shorts", True)
    short_entries = (tradable & base_short & htf_short_ok & rising & cross_down & pull_short) \
                    if allow_shorts else np.zeros(len(df), dtype=bool)

    if diag:
        def cnt(x): return int(np.count_nonzero(x))
        print("\n[DIAG] counts")
        print(f"  rows={len(df)}  tradable={cnt(tradable)}  gate={cnt(gate)}  session={cnt(sess)}")
        print(f"  base_long={cnt(base_long)}  base_short={cnt(base_short)}")
        print(f"  long_entries={cnt(long_entries)}  short_entries={cnt(short_entries)}")
        print(f"  rsi[min,max]={np.nanmin(indis['rsi']):.2f}, {np.nanmax(indis['rsi']):.2f}  "
              f"adx[min,max]={np.nanmin(indis['adx']):.2f}, {np.nanmax(indis['adx']):.2f}")
        atr_pct = indis["atr"] / np.where(close==0, np.nan, close)
        print(f"  atr/close[min,max]={np.nanmin(atr_pct):.5f}, {np.nanmax(atr_pct):.5f}")

    from .exits import HybridExit
    exit_logic = HybridExit(
        sl_atr_mult=params.get("sl_k",2.6),
        rr=params.get("rr",3.0),
        atr_trail_mult=params.get("trail_k",1.0),
        time_stop_bars=params.get("time_stop",48),
        fee_bps=fee_bps,
        be_after_r=params.get("be_after_r", 1.0),
        use_trailing=True,
        tp1_r=params.get("tp1_r", None),
        tp1_frac=params.get("tp1_frac", 0.5),
        intrabar_policy=params.get("intrabar_policy", "conservative"),
    )


    equity = initial_balance
    equity_curve = [equity]
    realized_equity_curve = [equity]
    wins = 0; losses = 0
    trade_returns = []
    gross_trade_pnls = []
    net_trade_pnls = []
    total_fees = 0.0
    total_funding = 0.0
    consec_losses = 0
    cb_max = params.get("cb_max_losses", 4)
    cb_cool = params.get("cb_cool_bars", 48)
    cooldown_until = -1

    i = 0
    n = len(df)
    while i < n-1:
        if cooldown_until >= 0 and i >= cooldown_until:
            cooldown_until = -1
            consec_losses = 0
        if cooldown_until >= 0 and i < cooldown_until:
            equity_curve.append(equity); realized_equity_curve.append(equity); i += 1; continue

        if long_entries[i] or short_entries[i]:
            direction = 1 if long_entries[i] else -1
            entry_idx = i + max(0, entry_delay_bars)
            if entry_idx >= n:
                equity_curve.append(equity); realized_equity_curve.append(equity); i += 1; continue
            raw_entry_price = open_[entry_idx] if entry_delay_bars > 0 else close[i]
            entry_price = raw_entry_price * (1.0 + slippage_rate) if direction == 1 else raw_entry_price * (1.0 - slippage_rate)
            atr_i = indis["atr"][i]
            sl_dist = exit_logic.sl_k * atr_i
            if sl_dist <= 0 or np.isnan(sl_dist):
                equity_curve.append(equity); realized_equity_curve.append(equity); i += 1; continue
            risk_amount = equity * risk_pct
            qty = max(risk_amount / sl_dist, 0.0)

            exit_result = exit_logic.apply(
                entry_idx,
                direction,
                entry_price,
                indis["atr"],
                high,
                low,
                close,
                first_bar_offset=0 if entry_delay_bars > 0 else 1,
                sl_dist_override=sl_dist,
                return_details=True,
            )
            exit_price = exit_result.blended_exit_price()
            exit_price = exit_price * (1.0 - slippage_rate) if direction == 1 else exit_price * (1.0 + slippage_rate)
            gross_pnl = (exit_price - entry_price) * qty if direction == 1 else (entry_price - exit_price) * qty

            entry_notional = abs(entry_price * qty)
            exit_notional = abs(exit_price * qty)
            fees = (entry_notional + exit_notional) * fee_rate
            bars_held = max(0, int(exit_result.exit_index) - int(entry_idx) + 1)
            funding = entry_notional * funding_rate_per_8h * (bars_held / 8.0) * direction
            pnl = gross_pnl - fees - funding

            entry_fee = entry_notional * fee_rate
            for k in range(entry_idx, int(exit_result.exit_index)):
                adverse_price = low[k] if direction == 1 else high[k]
                adverse_price = adverse_price * (1.0 - slippage_rate) if direction == 1 else adverse_price * (1.0 + slippage_rate)
                unrealized = (adverse_price - entry_price) * qty if direction == 1 else (entry_price - adverse_price) * qty
                held = max(1, k - entry_idx + 1)
                funding_so_far = entry_notional * funding_rate_per_8h * (held / 8.0) * direction
                exit_fee_est = abs(adverse_price * qty) * fee_rate
                equity_curve.append(equity + unrealized - entry_fee - exit_fee_est - funding_so_far)
                realized_equity_curve.append(equity)

            equity += pnl
            equity_curve.append(equity)
            realized_equity_curve.append(equity)
            if pnl > 0:
                wins += 1; consec_losses = 0
            else:
                losses += 1; consec_losses += 1
                if consec_losses >= cb_max:
                    cooldown_until = max(cooldown_until, int(exit_result.exit_index) + cb_cool)
            trade_returns.append(pnl / initial_balance)
            gross_trade_pnls.append(gross_pnl)
            net_trade_pnls.append(pnl)
            total_fees += fees
            total_funding += funding
            i = max(i + 1, int(exit_result.exit_index) + 1)
        else:
            equity_curve.append(equity)
            realized_equity_curve.append(equity)
            i += 1

    equity_curve = np.array(equity_curve, dtype=np.float64)
    rpt = summarize_report(
        initial_balance,
        equity_curve,
        np.array(trade_returns),
        wins,
        losses,
        realized_equity_curve=np.array(realized_equity_curve, dtype=np.float64),
        gross_trade_pnls=np.array(gross_trade_pnls, dtype=np.float64),
        net_trade_pnls=np.array(net_trade_pnls, dtype=np.float64),
        total_fees=total_fees,
        total_funding=total_funding,
    )
    return rpt, equity_curve

# -------------------- 스윕 & CLI --------------------
def coarse_grid() -> list[dict]:
    grid = []
    for min_adx in [12.0, 15.0, 20.0]:
        for min_atr_pct in [0.0010, 0.0015, 0.0020]:
            for rr in [2.0, 2.5, 3.0]:
                for time_stop in [24, 36, 48]:
                    grid.append({
                        "ema_short":14, "ema_long":100, "rsi_n":21, "adx_n":14, "atr_n":21,
                        "sl_k":2.6, "rr":rr, "trail_k":1.0, "time_stop":time_stop,
                        "min_adx":min_adx, "min_atr_pct":min_atr_pct,
                        "long_rsi_low":52, "long_rsi_high":60,
                        "short_rsi_low":40, "short_rsi_high":48,
                        "allow_hours": list(range(9,24)),         # 기본 시간대 약간 제한
                        "cb_max_losses": 4, "cb_cool_bars": 48,
                        "use_htf": True, "htf":"4H", "htf_ema_short":14, "htf_ema_long":100, "htf_adx_n":14, "htf_min_adx":12.0,
                        "require_adx_rising": True, "min_adx_slope": 0.0,
                        "use_rsi_cross": True, "entry_pullback_k": 0.5,
                    })
    return grid

def dense_grid() -> list[dict]:
    grid = []
    # Focused dense grid for recent low-ATR regime with HTF on and long-only
    for min_atr_pct in [0.0008, 0.0015]:
        for rr in [3.5, 4.0]:
            for time_stop in [36, 48]:
                for pull_k in [0.4, 0.6]:
                    for rsi_low in [48, 50]:
                        for use_rsi_band in [True, False]:
                            grid.append({
                                "ema_short":14, "ema_long":100, "rsi_n":21, "adx_n":14, "atr_n":21,
                                "sl_k":2.6, "rr":rr, "trail_k":0.8, "time_stop":time_stop,
                                "min_adx":10.0, "min_atr_pct":min_atr_pct,
                                "long_rsi_low":rsi_low, "long_rsi_high":100,
                                "short_rsi_low":40, "short_rsi_high":48,
                                "allow_hours": list(range(0,24)),
                                "cb_max_losses": 4, "cb_cool_bars": 48,
                                "use_htf": True, "htf":"4H", "htf_ema_short":14, "htf_ema_long":100, "htf_adx_n":14, "htf_min_adx":10.0,
                                "require_adx_rising": False, "min_adx_slope": 0.0,
                                "use_rsi_cross": True, "entry_pullback_k": pull_k,
                                "be_after_r": 0.8,
                                "use_cross": True, "use_rsi_band": use_rsi_band,
                                "allow_shorts": False,
                            })
    return grid

def run_param_sweep(df: pd.DataFrame, out_json: str | None = None, topk: int = 20):
    grid = coarse_grid()
    results = []
    for p in grid:
        rpt, _ = simulate(df, p, initial_balance=10_000.0, risk_pct=0.02, fee_bps=2.0, no_gate=False, diag=False)
        rpt["Param"] = p
        results.append(rpt)
    res_df = pd.DataFrame(results).sort_values("Total Net Pnl Percentage", ascending=False).reset_index(drop=True)
    if out_json:
        payload = res_df.head(topk).to_dict(orient="records")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n--- 최종 결과 요약 (상위 {}) ---".format(min(topk, len(res_df))))
    print(res_df.head(topk)[["Total Net Pnl Percentage","Num Trades","Win Rate Percentage","Profit Factor","Max Drawdown Percentage"]])
    return res_df

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EMA/RSI/ATR v2 Backtester (HTF+ADX rising+RSI cross+pullback)")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--grid", type=str, default="coarse", choices=["coarse","dense"], help="Sweep grid preset")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--no_gate", action="store_true")
    # Date range filters (UTC)
    ap.add_argument("--start", type=str, default=None, help="Start datetime (e.g., 2025-05-01 or 2025-05-01T00:00:00Z)")
    ap.add_argument("--end", type=str, default=None, help="End datetime (inclusive; e.g., 2025-08-13 or 2025-08-13T23:59:59Z)")

    # Strategy params
    ap.add_argument("--ema_short", type=int, default=14)
    ap.add_argument("--ema_long", type=int, default=100)
    ap.add_argument("--rsi_n", type=int, default=21)
    ap.add_argument("--adx_n", type=int, default=14)
    ap.add_argument("--atr_n", type=int, default=21)

    ap.add_argument("--sl_k", type=float, default=2.6)
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--trail_k", type=float, default=1.0)
    ap.add_argument("--time_stop", type=int, default=48)
    # Partial take-profit (TP1)
    ap.add_argument("--tp1_r", type=float, default=None, help="First take-profit at R multiple; None to disable")
    ap.add_argument("--tp1_frac", type=float, default=0.5, help="Fraction to scale out at TP1 (0-1)")

    ap.add_argument("--min_adx", type=float, default=15.0)
    ap.add_argument("--min_atr_pct", type=float, default=0.002)

    ap.add_argument("--long_rsi_low", type=float, default=52.0)
    ap.add_argument("--long_rsi_high", type=float, default=60.0)
    ap.add_argument("--short_rsi_low", type=float, default=40.0)
    ap.add_argument("--short_rsi_high", type=float, default=48.0)

    ap.add_argument("--hours", type=str, default="9-23")
    ap.add_argument("--cb_max_losses", type=int, default=4)
    ap.add_argument("--cb_cool_bars", type=int, default=48)

    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=2.0)
    ap.add_argument("--fee_bps", type=float, default=2.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--funding_rate_per_8h", type=float, default=0.0)
    ap.add_argument("--entry_delay_bars", type=int, default=1, help="1 = next-bar open fill; 0 = signal-close fill")
    ap.add_argument("--intrabar_policy", choices=["conservative", "optimistic"], default="conservative")

    # New toggles
    ap.add_argument("--use_htf", action="store_true")
    ap.add_argument("--htf", type=str, default="4H")
    ap.add_argument("--htf_ema_short", type=int, default=14)
    ap.add_argument("--htf_ema_long", type=int, default=100)
    ap.add_argument("--htf_adx_n", type=int, default=14)
    ap.add_argument("--htf_min_adx", type=float, default=12.0)

    ap.add_argument("--require_adx_rising", action="store_true")
    ap.add_argument("--min_adx_slope", type=float, default=0.0)

    ap.add_argument("--use_rsi_cross", action="store_true")
    ap.add_argument("--entry_pullback_k", type=float, default=0.0)
    ap.add_argument("--be_after_r", type=float, default=1.0, help="R multiple to move stop to break-even; <=0 to disable")
    ap.add_argument("--no_shorts", action="store_true", help="disable short entries (long-only)")
    # New entry flexibility
    ap.add_argument("--use_cross", action="store_true", help="Require EMA cross for entries (default True)")
    ap.add_argument("--no_cross", action="store_true", help="Do NOT require EMA cross (trend-follow)")
    ap.add_argument("--use_rsi_band", action="store_true", help="Use RSI band filters (default True)")
    ap.add_argument("--no_rsi_band", action="store_true", help="Disable RSI band filters")

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    df = load_csv(args.csv)
    # Optional date filtering (timestamps assumed UTC)
    if args.start or args.end:
        start_ts = pd.to_datetime(args.start, utc=True) if args.start else None
        end_ts = pd.to_datetime(args.end, utc=True) if args.end else None
        if start_ts is not None:
            df = df[df["timestamp"] >= start_ts]
        if end_ts is not None:
            df = df[df["timestamp"] <= end_ts]
        df = df.reset_index(drop=True)
    params = {
        "ema_short": args.ema_short, "ema_long": args.ema_long,
        "rsi_n": args.rsi_n, "adx_n": args.adx_n, "atr_n": args.atr_n,
    "sl_k": args.sl_k, "rr": args.rr, "trail_k": args.trail_k, "time_stop": args.time_stop,
    "tp1_r": args.tp1_r, "tp1_frac": args.tp1_frac,
        "min_adx": args.min_adx, "min_atr_pct": args.min_atr_pct,
        "long_rsi_low": args.long_rsi_low, "long_rsi_high": args.long_rsi_high,
        "short_rsi_low": args.short_rsi_low, "short_rsi_high": args.short_rsi_high,
        "allow_hours": parse_hours(args.hours),
        "cb_max_losses": args.cb_max_losses, "cb_cool_bars": args.cb_cool_bars,
        "use_htf": args.use_htf, "htf": args.htf,
        "htf_ema_short": args.htf_ema_short, "htf_ema_long": args.htf_ema_long,
        "htf_adx_n": args.htf_adx_n, "htf_min_adx": args.htf_min_adx,
        "require_adx_rising": args.require_adx_rising, "min_adx_slope": args.min_adx_slope,
    "use_rsi_cross": args.use_rsi_cross, "entry_pullback_k": args.entry_pullback_k,
        "be_after_r": args.be_after_r,
    "allow_shorts": (not args.no_shorts),
    "use_cross": (False if args.no_cross else True),
    "use_rsi_band": (False if args.no_rsi_band else True),
    "entry_delay_bars": args.entry_delay_bars,
    "slippage_bps": args.slippage_bps,
    "funding_rate_per_8h": args.funding_rate_per_8h,
    "intrabar_policy": args.intrabar_policy,

    }
    if args.sweep:
        # Choose grid
        if args.grid == "dense":
            grid = dense_grid()
        else:
            grid = coarse_grid()
        results = []
        for p in grid:
            rpt, _ = simulate(df, p, initial_balance=10_000.0, risk_pct=0.02, fee_bps=2.0, no_gate=False, diag=False)
            rpt["Param"] = p
            results.append(rpt)
        res_df = pd.DataFrame(results).sort_values("Total Net Pnl Percentage", ascending=False).reset_index(drop=True)
        if args.out:
            payload = res_df.head(20).to_dict(orient="records")
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print("\n--- 최종 결과 요약 (상위 {}) ---".format(min(20, len(res_df))))
        print(res_df.head(20)[["Total Net Pnl Percentage","Num Trades","Win Rate Percentage","Profit Factor","Max Drawdown Percentage"]])
    else:
        rpt, _ = simulate(df, params,
                          initial_balance=args.initial,
                          risk_pct=args.risk/100.0,
                          fee_bps=args.fee_bps,
                          no_gate=args.no_gate,
                          diag=args.diag)
        print(json.dumps(rpt, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
