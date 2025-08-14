from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Requires columns: 'high', 'low', 'close'.
    Returns a pd.Series indexed like df with ATR values.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.rename("atr")


@dataclass
class VolatilityFilterConfig:
    atr_period: int = 14
    min_move_pct: float = 2.0  # percent threshold; passes if (ATR/price)*100 >= threshold


def passes_vol_filter(
    df: pd.DataFrame,
    idx: int | pd.Timestamp,
    min_move_pct: float = 2.0,
    atr_period: int = 14,
    price_col: str = "close",
) -> bool:
    """
    Check if volatility conditions are met at index `idx`.

    Logic: ATR as % of price >= min_move_pct.

    Parameters
    - df: DataFrame with ['high','low','close'] (price_col can override close).
    - idx: row index (positional or datetime index label).
    - min_move_pct: threshold in percent.
    - atr_period: ATR period.

    Returns
    - bool indicating if the bar passes the filter.
    """
    if "atr" in df.columns:
        atr = df.loc[idx, "atr"]
    else:
        atr = compute_atr(df, atr_period).loc[idx]

    price = float(df.loc[idx, price_col])
    if not np.isfinite(atr) or not np.isfinite(price) or price <= 0:
        return False

    atr_pct = (atr / price) * 100.0
    return bool(atr_pct >= float(min_move_pct))


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Return a copy of df with an 'atr' column computed."""
    out = df.copy()
    out["atr"] = compute_atr(out, period)
    return out


def compute_atr_percent(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
    """
    Compute ATR as percent of price: (ATR/price)*100.
    Returns a Series named 'atr_percent'.
    """
    atr = compute_atr(df, period)
    price = df[price_col].astype(float)
    atr_pct = (atr / price) * 100.0
    return atr_pct.rename("atr_percent")


def add_atr_percent(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
    """Return a copy of df with an 'atr_percent' column computed."""
    out = df.copy()
    out["atr_percent"] = compute_atr_percent(out, period, price_col)
    return out
