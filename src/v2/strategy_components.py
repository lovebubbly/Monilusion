# -*- coding: utf-8 -*-
# src/v2/strategy_components.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1 or arr.size == 0:
        return arr.astype(np.float64, copy=True)
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    # 초기값: NaN이면 0으로
    first = arr[0]
    if np.isnan(first):
        first = 0.0
    out[0] = first
    for i in range(1, len(arr)):
        x = arr[i]
        if np.isnan(x):
            # 입력이 NaN이면 '직전 out'을 유지 (NaN 전파 방지)
            out[i] = out[i-1]
        else:
            prev = out[i-1]
            if np.isnan(prev):
                prev = x  # 이론상 여기 안 올 테지만 안전장치
            out[i] = alpha * x + (1 - alpha) * prev
    return out

def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's RMA (smoothed moving average), NaN-safe
    """
    out = np.empty_like(arr, dtype=np.float64)
    out[:] = 0.0
    if period <= 1 or arr.size == 0:
        return arr.astype(np.float64, copy=True)

    # 초기값 = 첫 period 구간 평균
    window = min(period, len(arr))
    first = np.nanmean(arr[:window])
    if np.isnan(first):
        first = 0.0
    out[window-1] = first

    alpha = 1.0 / period
    prev = first
    for i in range(window, len(arr)):
        x = arr[i]
        if np.isnan(x):
            x = 0.0
        prev = prev + alpha * (x - prev)
        out[i] = prev

    # 앞부분(0..window-2)은 0으로 두거나 자연스레 채우기
    for i in range(0, window-1):
        out[i] = out[window-1]
    return out



def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ema_gain = _ema(gain, period)
    ema_loss = _ema(loss, period)
    rs = np.divide(ema_gain, ema_loss, out=np.zeros_like(ema_gain), where=(ema_loss != 0))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return tr

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = true_range(high, low, close)
    return _ema(tr, period)

def dmi_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    # +DM, -DM
    up_move = high - np.roll(high, 1); up_move[0] = 0.0
    down_move = np.roll(low, 1) - low; down_move[0] = 0.0

    plus_dm  = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
    minus_dm = np.where((down_move > 0) & (down_move > up_move), down_move, 0.0)

    # TR
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    # Wilder RMA
    atr_val   = _rma(tr,        period)
    plus_dm_s = _rma(plus_dm,   period)
    minus_dm_s= _rma(minus_dm,  period)

    # 분모 안전화
    eps = 1e-12
    atr_safe = np.where(atr_val <= 0, 1.0, atr_val)

    plus_di  = 100.0 * (plus_dm_s  / (atr_safe + eps))
    minus_di = 100.0 * (minus_dm_s / (atr_safe + eps))

    denom = plus_di + minus_di
    denom_safe = np.where(denom <= 0, 1.0, denom)
    dx = 100.0 * np.abs(plus_di - minus_di) / (denom_safe + eps)

    # ADX = DX의 Wilder RMA
    adx = _rma(dx, period)

    # NaN/Inf 제거
    plus_di  = np.nan_to_num(plus_di,  nan=0.0, posinf=0.0, neginf=0.0)
    minus_di = np.nan_to_num(minus_di, nan=0.0, posinf=0.0, neginf=0.0)
    adx      = np.nan_to_num(adx,      nan=0.0, posinf=0.0, neginf=0.0)
    return plus_di, minus_di, adx


def ema_crossover_signals(close: np.ndarray, short_n: int, long_n: int):
    ema_s = _ema(close, short_n)
    ema_l = _ema(close, long_n)
    above = ema_s > ema_l
    cross_up = (above & ~np.roll(above, 1))
    cross_down = (~above & np.roll(above, 1))
    cross_up[0] = False
    cross_down[0] = False
    return ema_s, ema_l, cross_up, cross_down

def session_mask(timestamps: pd.Series, allowed_hours: list[int]) -> np.ndarray:
    # timestamps: pandas Series (UTC assumed) -> hours in 0..23 allowed list
    hours = timestamps.dt.hour.values
    if not allowed_hours:
        return np.ones(len(hours), dtype=bool)
    allowed = np.zeros(len(hours), dtype=bool)
    for h in allowed_hours:
        allowed |= (hours == h)
    return allowed

def build_indicators(df: pd.DataFrame,
                     ema_short=14, ema_long=100, rsi_n=21, adx_n=14, atr_n=21):
    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    ema_s, ema_l, xup, xdn = ema_crossover_signals(close, ema_short, ema_long)
    r = rsi(close, rsi_n)
    a = atr(high, low, close, atr_n)
    plus_di, minus_di, adx = dmi_adx(high, low, close, adx_n)
    # build_indicators 끝나기 직전 한 줄씩 추가 (선택)
    r = np.nan_to_num(r, nan=50.0)           # RSI 중심값 대체
    a = np.nan_to_num(a, nan=0.0)
    plus_di, minus_di, adx = dmi_adx(high, low, close, adx_n)
    plus_di = np.nan_to_num(plus_di, 0.0); minus_di = np.nan_to_num(minus_di, 0.0); adx = np.nan_to_num(adx, 0.0)

    return {
        "ema_s": ema_s, "ema_l": ema_l,
        "cross_up": xup, "cross_down": xdn,
        "rsi": r, "atr": a, "plus_di": plus_di, "minus_di": minus_di, "adx": adx
    }

def regime_gate(indis: dict,
                min_adx: float = 15.0,
                min_atr_pct: float = 0.002,
                close: np.ndarray | None = None) -> np.ndarray:
    adx = np.nan_to_num(indis["adx"], nan=0.0, posinf=0.0, neginf=0.0)
    adx_ok = adx >= min_adx
    if close is None:
        return adx_ok
    atr_pct = np.nan_to_num(indis["atr"] / np.where(close==0, np.nan, close), nan=0.0, posinf=0.0, neginf=0.0)
    vol_ok = atr_pct >= min_atr_pct
    return adx_ok & vol_ok


def long_entry_mask(df: pd.DataFrame, indis: dict,
                    rsi_low=52, rsi_high=60,
                    use_cross: bool = True,
                    use_rsi_band: bool = True) -> np.ndarray:
    rsi_ok = (indis["rsi"] >= rsi_low) & (indis["rsi"] <= rsi_high) if use_rsi_band else np.ones(len(df), dtype=bool)
    if use_cross:
        trend_ok = indis["cross_up"] & (indis["plus_di"] > indis["minus_di"]) & (indis["ema_s"] > indis["ema_l"])
    else:
        # Trend-follow without requiring immediate cross; DI and EMA alignment suffice
        trend_ok = (indis["plus_di"] > indis["minus_di"]) & (indis["ema_s"] > indis["ema_l"])
    return rsi_ok & trend_ok

def short_entry_mask(df: pd.DataFrame, indis: dict,
                     rsi_low=40, rsi_high=48,
                     use_cross: bool = True,
                     use_rsi_band: bool = True) -> np.ndarray:
    # 더 보수적인 Short: 여러 컨플루언스 필요
    rsi_ok = (indis["rsi"] >= rsi_low) & (indis["rsi"] <= rsi_high) if use_rsi_band else np.ones(len(df), dtype=bool)
    if use_cross:
        trend_ok = indis["cross_down"] & (indis["minus_di"] > indis["plus_di"]) & (indis["ema_s"] < indis["ema_l"])
    else:
        trend_ok = (indis["minus_di"] > indis["plus_di"]) & (indis["ema_s"] < indis["ema_l"])
    return rsi_ok & trend_ok
