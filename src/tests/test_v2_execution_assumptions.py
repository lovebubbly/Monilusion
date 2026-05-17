from __future__ import annotations

import numpy as np
import pandas as pd

from src.v2.backtest_v2 import build_htf_context
from src.v2.exits import HybridExit
from src.v2.metrics import summarize_report


def test_hybrid_exit_conservative_resolves_ambiguous_bar_to_stop():
    exit_logic = HybridExit(sl_atr_mult=1.0, rr=1.0, intrabar_policy="conservative")
    atr = np.array([1.0, 1.0], dtype=np.float64)
    high = np.array([101.5, 100.0], dtype=np.float64)
    low = np.array([98.5, 100.0], dtype=np.float64)
    close = np.array([100.0, 100.0], dtype=np.float64)

    result = exit_logic.apply(
        0, 1, 100.0, atr, high, low, close, first_bar_offset=0, return_details=True
    )

    assert result.exit_index == 0
    assert result.exit_price == 99.0
    assert result.reason == "SL_AMBIGUOUS"


def test_hybrid_exit_optimistic_resolves_ambiguous_bar_to_target():
    exit_logic = HybridExit(sl_atr_mult=1.0, rr=1.0, intrabar_policy="optimistic")
    atr = np.array([1.0, 1.0], dtype=np.float64)
    high = np.array([101.5, 100.0], dtype=np.float64)
    low = np.array([98.5, 100.0], dtype=np.float64)
    close = np.array([100.0, 100.0], dtype=np.float64)

    result = exit_logic.apply(
        0, 1, 100.0, atr, high, low, close, first_bar_offset=0, return_details=True
    )

    assert result.exit_index == 0
    assert result.exit_price == 101.0
    assert result.reason == "TP_AMBIGUOUS"


def test_htf_context_does_not_backfill_future_values_into_warmup():
    ts = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    close = np.linspace(100.0, 148.0, len(ts))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }
    )

    long_ok, short_ok = build_htf_context(df, htf="4h", ema_short=2, ema_long=3, adx_n=2, min_adx=0.0)

    assert not long_ok[:4].any()
    assert not short_ok[:4].any()


def test_summarize_report_separates_gross_and_net_profit_factor():
    equity_curve = np.array([10_000.0, 10_100.0, 10_050.0], dtype=np.float64)

    report = summarize_report(
        10_000.0,
        equity_curve,
        np.array([100.0, -50.0], dtype=np.float64),
        1,
        1,
        gross_trade_pnls=np.array([120.0, -40.0], dtype=np.float64),
        net_trade_pnls=np.array([100.0, -50.0], dtype=np.float64),
        total_fees=10.0,
        total_funding=5.0,
    )

    assert report["Profit Factor"] == 2.0
    assert report["Net Profit Factor"] == 2.0
    assert report["Gross Profit Factor"] == 3.0
    assert report["Total Fees"] == 10.0
    assert report["Total Funding"] == 5.0
