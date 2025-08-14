from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, +DI, -DI. Requires 'high','low','close'.
    Returns DataFrame with columns ['plus_di','minus_di','adx'].
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(span=period, adjust=False).mean()

    out = pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})
    return out


def passes_regime_filter(
    df: pd.DataFrame,
    idx: int | pd.Timestamp,
    adx_min: float = 25.0,
    ema_fast: int = 10,
    ema_slow: int = 50,
    side: Literal["Long", "Short", "Both"] = "Both",
    price_col: str = "close",
) -> bool:
    """
    Regime gate using ADX and EMA slope/relationship.

    Conditions:
    - ADX >= adx_min
    - For Long: EMA(fast) > EMA(slow)
      For Short: EMA(fast) < EMA(slow)
      If side == 'Both': only ADX condition is enforced.
    """
    price = df[price_col].astype(float)
    f = _ema(price, ema_fast)
    s = _ema(price, ema_slow)

    adx_df = compute_adx(df)
    adx = adx_df["adx"]

    try:
        adx_ok = float(adx.loc[idx]) >= adx_min
        if side == "Long":
            trend_ok = float(f.loc[idx]) > float(s.loc[idx])
        elif side == "Short":
            trend_ok = float(f.loc[idx]) < float(s.loc[idx])
        else:
            trend_ok = True
        return bool(adx_ok and trend_ok)
    except Exception:
        return False


def is_favorable_regime(
    adx_value: float,
    atr_percent_value: float,
    adx_threshold: float = 25.0,
    atr_percent_threshold: float = 2.0,
) -> bool:
    """
    Return True if both trend strength (ADX) and volatility (ATR%) meet thresholds.
    """
    try:
        return (float(adx_value) >= float(adx_threshold)) and (
            float(atr_percent_value) >= float(atr_percent_threshold)
        )
    except Exception:
        return False


def favorable_regime_from_df(
    df: pd.DataFrame,
    idx: int | pd.Timestamp,
    adx_period: int = 14,
    atr_period: int = 14,
    adx_threshold: float = 25.0,
    atr_percent_threshold: float = 2.0,
    price_col: str = "close",
) -> bool:
    """
    Convenience wrapper: compute ADX and ATR% from DataFrame, then apply is_favorable_regime.
    Requires columns: 'high','low','close'.
    """
    try:
        adx_df = compute_adx(df, adx_period)
        adx_val = adx_df.loc[idx, "adx"]
        from . import volatility_filter as _vf  # type: ignore

        atr_pct = _vf.compute_atr_percent(df, atr_period, price_col)
        atr_pct_val = atr_pct.loc[idx]
        return is_favorable_regime(adx_val, atr_pct_val, adx_threshold, atr_percent_threshold)
    except Exception:
        return False
