from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import diff_cuda_cpu_reference as ref  # noqa: E402


def _frame(*, open_at_entry: float = 100.0, adverse_low: float = 99.0) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=8, freq="h")
    close = np.full(len(ts), 100.0, dtype=np.float64)
    open_ = close.copy()
    open_[3] = open_at_entry
    high = np.full(len(ts), 101.0, dtype=np.float64)
    low = np.full(len(ts), 99.0, dtype=np.float64)
    low[4] = adverse_low
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(len(ts), 1.0, dtype=np.float64),
        },
        index=ts,
    )


def _indicator_stub(df: pd.DataFrame, params: dict, htf: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    ema_short = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    ema_long = np.ones(n, dtype=np.float64)
    return {
        "ema_short": ema_short,
        "ema_long": ema_long,
        "atr_sl": np.full(n, float(params.get("test_atr", 50.0)), dtype=np.float64),
        "atr_trail": np.full(n, float(params.get("test_atr", 50.0)), dtype=np.float64),
        "vol_sma": np.zeros(n, dtype=np.float64),
        "rsi": np.full(n, 60.0, dtype=np.float64),
        "adx": np.full(n, 50.0, dtype=np.float64),
        "h4_ema": np.full(n, 90.0, dtype=np.float64),
        "h4_adx": np.full(n, 50.0, dtype=np.float64),
        "h4_atr": np.full(n, 1.0, dtype=np.float64),
    }


def _params() -> dict:
    return {
        "ema_short_h1": 2,
        "ema_long_h1": 3,
        "allow_short_entries": False,
        "risk_per_trade_percentage": 0.01,
        "atr_multiplier_sl": 1.0,
        "risk_reward_ratio": 10.0,
        "time_stop_period_hours": 100,
        "exit_strategy_type": ref.EXIT_FIXED_RR,
        "test_atr": 50.0,
    }


def test_htf_resample_alignment_does_not_expose_incomplete_4h_bar():
    ts = pd.date_range("2026-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "Open": np.arange(10, dtype=np.float64),
            "High": np.arange(10, dtype=np.float64),
            "Low": np.arange(10, dtype=np.float64),
            "Close": np.arange(10, dtype=np.float64),
            "Volume": np.ones(10, dtype=np.float64),
        },
        index=ts,
    )

    htf = ref._resample_htf(df, "4h")
    aligned = ref._align_htf(htf, df.index)

    assert pd.isna(aligned.loc[ts[3], "Close"])
    assert aligned.loc[ts[4], "Close"] == 3.0
    assert aligned.loc[ts[7], "Close"] == 3.0
    assert aligned.loc[ts[8], "Close"] == 7.0


def test_cpu_reference_entry_uses_next_bar_open_not_signal_close():
    original = ref._indicators
    ref._indicators = _indicator_stub
    try:
        metrics, events = ref.cpu_reference_backtest(
            _frame(open_at_entry=123.0),
            _params(),
            initial_balance=10_000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            entry_delay_bars=1,
            funding_rate_per_8h=0.0,
            include_entry_events=True,
        )
    finally:
        ref._indicators = original

    accepted = [row for row in events if row["status"] == "accepted"]
    assert accepted
    assert accepted[0]["entry_signal_index"] == 2
    assert accepted[0]["entry_index"] == 3
    assert accepted[0]["theoretical_entry_price"] == 123.0
    assert metrics["num_trades"] == 0


def test_cpu_reference_mdd_uses_intrabar_adverse_extreme_not_close_only():
    original = ref._indicators
    ref._indicators = _indicator_stub
    try:
        metrics = ref.cpu_reference_backtest(
            _frame(open_at_entry=100.0, adverse_low=80.0),
            _params(),
            initial_balance=10_000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            entry_delay_bars=1,
            funding_rate_per_8h=0.0,
        )
    finally:
        ref._indicators = original

    assert metrics["max_drawdown_percentage"] == 0.4


def test_cpu_cuda_diff_requires_gross_and_net_profit_factor_match():
    diff = ref._diff(
        {
            "total_net_pnl_percentage": 10.0,
            "num_trades": 2,
            "win_rate_percentage": 50.0,
            "net_profit_factor": 2.0,
            "gross_profit_factor": 3.0,
            "max_drawdown_percentage": 4.0,
        },
        {
            "total_net_pnl_percentage": 10.0,
            "num_trades": 2,
            "win_rate_percentage": 50.0,
            "net_profit_factor": 2.0,
            "gross_profit_factor": 2.5,
            "max_drawdown_percentage": 4.0,
        },
    )

    assert diff["net_profit_factor"]["abs_diff"] == 0.0
    assert diff["gross_profit_factor"]["abs_diff"] == 0.5
    assert not ref._diff_pass(diff, pnl_tol=0.01, pf_tol=0.01, mdd_tol=0.01, trade_tol=0)
