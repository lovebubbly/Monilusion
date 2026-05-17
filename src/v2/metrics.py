# -*- coding: utf-8 -*-
# src/v2/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(-dd.min() * 100.0) if len(equity_curve) else 0.0

def profit_factor(trade_returns: np.ndarray) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    return float(gains / losses)

def summarize_report(initial_balance: float,
                     equity_curve: np.ndarray,
                     trade_returns: np.ndarray,
                     wins: int, losses: int,
                     *,
                     realized_equity_curve: np.ndarray | None = None,
                     gross_trade_pnls: np.ndarray | None = None,
                     net_trade_pnls: np.ndarray | None = None,
                     total_fees: float = 0.0,
                     total_funding: float = 0.0) -> dict:
    final_balance = float(equity_curve[-1]) if len(equity_curve) else initial_balance
    pnl = final_balance - initial_balance
    pnl_pct = (pnl / initial_balance) * 100.0 if initial_balance else 0.0
    num_trades = wins + losses
    wr = (wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    net_pnls = np.array(net_trade_pnls if net_trade_pnls is not None else trade_returns, dtype=np.float64)
    gross_pnls = np.array(gross_trade_pnls if gross_trade_pnls is not None else trade_returns, dtype=np.float64)
    pf = profit_factor(net_pnls)
    gross_pf = profit_factor(gross_pnls)
    mdd_pct = max_drawdown(np.array(equity_curve, dtype=np.float64))
    realized_mdd_pct = (
        max_drawdown(np.array(realized_equity_curve, dtype=np.float64))
        if realized_equity_curve is not None
        else mdd_pct
    )
    return {
        "Initial Balance": initial_balance,
        "Final Balance": final_balance,
        "Total Net Pnl": pnl,
        "Total Net Pnl Percentage": pnl_pct,
        "Num Trades": num_trades,
        "Num Wins": wins,
        "Num Losses": losses,
        "Win Rate Percentage": wr,
        "Profit Factor": pf,
        "Net Profit Factor": pf,
        "Gross Profit Factor": gross_pf,
        "Max Drawdown Percentage": mdd_pct,
        "Max Intrabar Drawdown Percentage": mdd_pct,
        "Max Realized Drawdown Percentage": realized_mdd_pct,
        "Total Fees": float(total_fees),
        "Total Funding": float(total_funding),
    }
