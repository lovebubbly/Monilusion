from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta


POSITION_NONE = 0
POSITION_LONG = 1
POSITION_SHORT = -1
EXIT_FIXED_RR = "FixedRR"
EXIT_TRAILING_ATR = "TrailingATR"


def _num(value: Any) -> float:
    if value == "inf":
        return float("inf")
    return float(value)


def _parse_ts(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        unit = "ms"
        finite = numeric.dropna()
        if not finite.empty:
            max_abs = finite.abs().max()
            if max_abs >= 1e14:
                unit = "us"
            elif max_abs < 1e11:
                unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce").dt.tz_convert(None)
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(None)


def _load_ohlcv(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "Open time" not in df.columns:
        if "timestamp" in cols:
            df = df.rename(columns={cols["timestamp"]: "Open time"})
        elif "date" in cols:
            df = df.rename(columns={cols["date"]: "Open time"})
    rename_map = {}
    for want in ["Open", "High", "Low", "Close", "Volume"]:
        low = want.lower()
        if want not in df.columns and low in cols:
            rename_map[cols[low]] = want
    if rename_map:
        df = df.rename(columns=rename_map)
    required = ["Open time", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing OHLCV columns: {missing}")
    df = df[required].copy()
    df["Open time"] = _parse_ts(df["Open time"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["Open time"]).sort_values("Open time")
    df = df.set_index("Open time")

    start_ts = pd.to_datetime(start, utc=True).tz_convert(None)
    end_ts = pd.to_datetime(end, utc=True).tz_convert(None)
    if len(str(end).strip()) == 10 and str(end).count("-") == 2:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return df[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def load_funding_cumulative(df: pd.DataFrame, funding_csv: Path | None) -> np.ndarray | None:
    if funding_csv is None:
        return None
    path = funding_csv if funding_csv.is_absolute() else Path.cwd() / funding_csv
    if not path.exists():
        raise SystemExit(f"Funding CSV not found: {path}")
    frame = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in frame.columns}
    time_col = cols.get("funding_time") or cols.get("timestamp") or cols.get("time")
    rate_col = cols.get("funding_rate") or cols.get("fundingrate")
    if time_col is None or rate_col is None:
        raise SystemExit(f"Funding CSV must contain funding_time/timestamp and funding_rate columns: {path}")
    events = pd.DataFrame(
        {
            "timestamp": _parse_ts(frame[time_col]).dt.floor("h"),
            "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }
    ).dropna()
    if events.empty:
        return np.zeros(len(df), dtype=np.float64)
    hourly = events.groupby("timestamp")["funding_rate"].sum()
    aligned = hourly.reindex(df.index, fill_value=0.0).to_numpy(dtype=np.float64)
    return np.cumsum(aligned, dtype=np.float64)


def _recent_funding_rate(
    i: int,
    *,
    funding_cumulative: np.ndarray | None,
    funding_rate_per_8h: float,
    lookback_hours: int,
) -> float:
    lookback = max(1, int(lookback_hours))
    if funding_cumulative is None:
        return float(funding_rate_per_8h) * (lookback / 8.0)
    if i <= 0:
        return 0.0
    prev_idx = i - 1
    start_idx = max(0, prev_idx - lookback)
    return float(funding_cumulative[prev_idx] - funding_cumulative[start_idx])


def _volatility_target_multiplier(i: int, close: np.ndarray, params: dict[str, Any]) -> float:
    if not bool(params.get("use_volatility_target_sizing", False)):
        return 1.0
    target_annual = float(params.get("volatility_target_annual", 0.3))
    lookback_hours = int(params.get("volatility_lookback_hours", 168))
    min_mult = float(params.get("volatility_sizing_min_mult", 0.25))
    max_mult = float(params.get("volatility_sizing_max_mult", 2.0))
    if target_annual <= 0.0 or lookback_hours <= 1 or i <= 1:
        return 1.0

    effective_lookback = min(lookback_hours, i)
    weighted_var = 0.0
    weight_sum = 0.0
    weight = 1.0
    for k in range(effective_lookback):
        ret_idx = i - k
        if ret_idx <= 0:
            break
        prev_close = float(close[ret_idx - 1])
        curr_close = float(close[ret_idx])
        if prev_close > 0.0 and curr_close > 0.0:
            log_ret = math.log(curr_close / prev_close)
            weighted_var += weight * log_ret * log_ret
            weight_sum += weight
        weight *= 0.94

    if weight_sum <= 0.0:
        return 1.0
    forecast_annual = math.sqrt(weighted_var / weight_sum) * math.sqrt(8760.0)
    if forecast_annual <= 1e-12:
        return 1.0
    return min(max(target_annual / forecast_annual, min_mult), max_mult)


def _drawdown_sizing_multiplier(balance: float, peak_equity: float, params: dict[str, Any]) -> float:
    if not bool(params.get("use_drawdown_sizing", False)):
        return 1.0
    start_pct = float(params.get("drawdown_sizing_start_pct", 2.0))
    full_pct = float(params.get("drawdown_sizing_full_pct", 8.0))
    min_mult = float(params.get("drawdown_sizing_min_mult", 0.5))
    if peak_equity <= 0.0 or full_pct <= start_pct:
        return 1.0
    min_mult = max(0.01, min(1.0, min_mult))

    drawdown_pct = ((peak_equity - balance) / peak_equity) * 100.0
    if drawdown_pct <= start_pct:
        return 1.0
    if drawdown_pct >= full_pct:
        return min_mult

    progress = (drawdown_pct - start_pct) / (full_pct - start_pct)
    return max(min_mult, min(1.0, 1.0 - progress * (1.0 - min_mult)))


def _resample_htf(df: pd.DataFrame, htf: str) -> pd.DataFrame:
    return (
        df.resample(htf.lower(), label="right", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _align_htf(htf_df: pd.DataFrame, primary_index: pd.Index) -> pd.DataFrame:
    aligned = htf_df.reindex(primary_index, method="ffill")
    source_times = pd.Series(htf_df.index, index=htf_df.index).reindex(primary_index, method="ffill")
    valid = source_times.notna()
    if valid.any():
        primary_series = pd.Series(primary_index, index=primary_index)
        lookahead = source_times[valid] > primary_series[valid]
        if bool(lookahead.any()):
            raise RuntimeError(f"HTF lookahead detected at {lookahead[lookahead].index[0]}")
    return aligned


def _series_or_nan(series: pd.Series | None, n: int) -> np.ndarray:
    if series is None:
        return np.full(n, np.nan, dtype=np.float64)
    return series.to_numpy(dtype=np.float64)


def _indicators(df: pd.DataFrame, params: dict[str, Any], htf: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    adx_period = int(params.get("adx_period", 14))
    atr_sl_period = int(params.get("atr_period_sl", 14))
    atr_trail_period = int(params.get("trailing_atr_period", atr_sl_period))
    volume_sma_period = int(params.get("volume_sma_period", 20))
    rsi_period = int(params.get("rsi_period", 14))
    ema_htf_period = int(params.get("ema_htf", 50))

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=adx_period)
    htf_ema = ta.ema(htf["Close"], length=ema_htf_period)
    htf_adx_df = ta.adx(htf["High"], htf["Low"], htf["Close"], length=adx_period)
    return {
        "ema_short": _series_or_nan(ta.ema(df["Close"], length=int(params["ema_short_h1"])), n),
        "ema_long": _series_or_nan(ta.ema(df["Close"], length=int(params["ema_long_h1"])), n),
        "atr_sl": _series_or_nan(ta.atr(df["High"], df["Low"], df["Close"], length=atr_sl_period), n),
        "atr_trail": _series_or_nan(ta.atr(df["High"], df["Low"], df["Close"], length=atr_trail_period), n),
        "vol_sma": _series_or_nan(ta.sma(df["Volume"], length=volume_sma_period), n),
        "rsi": _series_or_nan(ta.rsi(df["Close"], length=rsi_period), n),
        "adx": _series_or_nan(adx_df.get(f"ADX_{adx_period}") if adx_df is not None else None, n),
        "h4_ema": _series_or_nan(htf_ema, n),
        "h4_adx": _series_or_nan(htf_adx_df.get(f"ADX_{adx_period}") if htf_adx_df is not None else None, n),
        "h4_atr": _series_or_nan(ta.atr(htf["High"], htf["Low"], htf["Close"], length=atr_sl_period), n),
    }


def _short_entry_ok(
    i: int,
    close: np.ndarray,
    low: np.ndarray,
    ind: dict[str, np.ndarray],
    params: dict[str, Any],
) -> bool:
    period = int(params.get("price_breakdown_period", 3))
    if i < period:
        return False
    trend_bearish = ind["ema_short"][i] < ind["ema_long"][i]
    adx_value = ind["adx"][i]
    trend_strong = bool(np.isnan(adx_value) or adx_value > float(params.get("adx_threshold_for_short", 25.0)))
    min_low = float(np.min(low[i - period : i]))
    price_breakdown = close[i] < min_low
    rsi = ind["rsi"][i]
    momentum = bool(np.isnan(rsi) or rsi < float(params.get("rsi_momentum_threshold", 40.0)))
    return bool(trend_bearish and trend_strong and price_breakdown and momentum)


def _profit_factor(values: list[float]) -> float:
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _safe_round(value: Any, digits: int = 8) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return round(out, digits)


def _entry_feature_snapshot(
    i: int,
    *,
    close: np.ndarray,
    ind: dict[str, np.ndarray],
    params: dict[str, Any],
    hour_of_day: np.ndarray,
    day_of_week: np.ndarray,
    ema_short: float,
    ema_long: float,
) -> dict[str, Any]:
    atr_sl = ind["atr_sl"][i]
    ema_spread_atr = abs(ema_short - ema_long) / atr_sl if atr_sl > 0 else np.nan
    h1_slope_log = np.nan
    h1_lb = int(params.get("h1_slope_lookback_bars", 24))
    if i - h1_lb >= 0 and ind["ema_long"][i - h1_lb] > 0:
        h1_slope_log = (ema_long - ind["ema_long"][i - h1_lb]) / ind["ema_long"][i - h1_lb]
    h4_slope_log = np.nan
    h4_curr_log = ind["h4_ema"][i]
    htf_lb = int(params.get("htf_slope_lookback_bars", 24))
    if i - htf_lb >= 0 and ind["h4_ema"][i - htf_lb] > 0:
        h4_slope_log = (h4_curr_log - ind["h4_ema"][i - htf_lb]) / ind["h4_ema"][i - htf_lb]
    return {
        "entry_hour_utc": int(hour_of_day[i]),
        "entry_day_of_week": int(day_of_week[i]),
        "entry_close": _safe_round(close[i]),
        "entry_adx": _safe_round(ind["adx"][i]),
        "entry_h4_adx": _safe_round(ind["h4_adx"][i]),
        "entry_rsi": _safe_round(ind["rsi"][i]),
        "entry_atr_pct": _safe_round((atr_sl / close[i]) * 100.0) if close[i] > 0 else None,
        "entry_h4_atr_pct": _safe_round((ind["h4_atr"][i] / close[i]) * 100.0) if close[i] > 0 else None,
        "entry_ema_spread_atr": _safe_round(ema_spread_atr),
        "entry_h1_slope_pct": _safe_round(h1_slope_log * 100.0),
        "entry_h4_slope_pct": _safe_round(h4_slope_log * 100.0),
    }


def cpu_reference_backtest(
    df: pd.DataFrame,
    params: dict[str, Any],
    *,
    initial_balance: float,
    commission_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
    funding_rate_per_8h: float,
    min_trade_size: float = 0.001,
    quantity_precision: int = 3,
    include_equity: bool = False,
    include_trades: bool = False,
    include_entry_events: bool = False,
    intrabar_policy: str = "conservative",
    funding_cumulative: np.ndarray | None = None,
) -> Any:
    if intrabar_policy not in {"conservative", "optimistic"}:
        raise ValueError("intrabar_policy must be one of: conservative, optimistic")
    if funding_cumulative is not None and len(funding_cumulative) != len(df):
        raise ValueError("funding_cumulative length must match df length")
    htf = _align_htf(_resample_htf(df, "4h"), df.index)
    ind = _indicators(df, params, htf)
    open_ = df["Open"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    hour_of_day = np.array([ts.hour for ts in df.index], dtype=np.int8)
    day_of_week = np.array([ts.weekday() for ts in df.index], dtype=np.int8)

    balance = float(initial_balance)
    position = POSITION_NONE
    entry_price = 0.0
    position_size = 0.0
    initial_sl = 0.0
    current_sl = 0.0
    take_profit = 0.0
    entry_idx = -1
    peak_equity = balance
    max_dd = 0.0
    gross_pnls: list[float] = []
    net_pnls: list[float] = []
    trades_detail: list[dict[str, Any]] = []
    entry_events: list[dict[str, Any]] = []
    equity_curve = [balance]
    wins = 0
    consecutive_losses = 0
    cooldown_until = -1
    dd_guard_until = -1
    entry_signal_idx = -1
    entry_features: dict[str, Any] = {}
    ambiguous_intrabar_exits = 0
    total_funding = 0.0

    def funding_cost_between(start_idx: int, end_idx: int, notional: float, direction: float) -> float:
        if funding_cumulative is None:
            held = max(1, end_idx - start_idx + 1)
            rate_sum = funding_rate_per_8h * (held / 8.0)
        else:
            if end_idx <= start_idx:
                rate_sum = 0.0
            else:
                rate_sum = float(funding_cumulative[end_idx] - funding_cumulative[start_idx])
        return notional * rate_sum * direction

    for i in range(1, len(df)):
        if position != POSITION_NONE:
            direction = 1.0 if position == POSITION_LONG else -1.0
            curr_profit = (
                (close[i] - entry_price) * position_size
                if position == POSITION_LONG
                else (entry_price - close[i]) * position_size
            )
            mtm_cost = (entry_price * position_size + close[i] * position_size) * (
                commission_rate + slippage_rate
            )
            funding_so_far = funding_cost_between(entry_idx, i, entry_price * position_size, direction)
            mark_equity = balance + curr_profit - mtm_cost - funding_so_far
            peak_equity = max(peak_equity, mark_equity)
            adverse = low[i] if position == POSITION_LONG else high[i]
            adverse_profit = (
                (adverse - entry_price) * position_size
                if position == POSITION_LONG
                else (entry_price - adverse) * position_size
            )
            adverse_cost = (entry_price * position_size + adverse * position_size) * (
                commission_rate + slippage_rate
            )
            adverse_equity = balance + adverse_profit - adverse_cost - funding_so_far
            dd = (peak_equity - adverse_equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd)
            equity_curve.append(adverse_equity)

            exit_price = None
            exit_reason = None
            intrabar_ambiguous_exit = False
            time_stop = int(params.get("time_stop_period_hours", 48))
            if i - entry_idx >= time_stop and curr_profit < 0:
                exit_price = close[i]
                exit_reason = "time_stop"

            if exit_price is None and bool(params.get("use_breakeven_stop", False)):
                risk_unit = abs(entry_price - initial_sl)
                if risk_unit > 0:
                    trigger = float(params.get("breakeven_trigger_r", 1.0))
                    offset = float(params.get("breakeven_offset_r", 0.0))
                    if position == POSITION_LONG:
                        if close[i] - entry_price >= risk_unit * trigger:
                            current_sl = max(current_sl, entry_price + risk_unit * offset)
                    elif entry_price - close[i] >= risk_unit * trigger:
                        current_sl = min(current_sl, entry_price - risk_unit * offset)

            if (
                exit_price is None
                and params.get("exit_strategy_type") == EXIT_TRAILING_ATR
                and curr_profit >= float(params.get("profit_threshold_for_trail", 1.0))
            ):
                atr_trail = ind["atr_trail"][i]
                if not np.isnan(atr_trail):
                    trail_mult = float(params.get("trailing_atr_multiplier", 1.5))
                    if position == POSITION_LONG:
                        current_sl = max(current_sl, high[i] - atr_trail * trail_mult)
                    else:
                        current_sl = min(current_sl, low[i] + atr_trail * trail_mult)

            if exit_price is None:
                fixed_rr = params.get("exit_strategy_type") == EXIT_FIXED_RR
                if position == POSITION_LONG:
                    stop_hit = low[i] <= current_sl
                    tp_hit = fixed_rr and high[i] >= take_profit
                    intrabar_ambiguous_exit = bool(stop_hit and tp_hit)
                    if intrabar_ambiguous_exit:
                        ambiguous_intrabar_exits += 1
                    if intrabar_policy == "optimistic" and intrabar_ambiguous_exit:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif stop_hit:
                        exit_price = current_sl
                        exit_reason = "stop_loss" if current_sl == initial_sl else "trailing_stop"
                    elif tp_hit:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                else:
                    stop_hit = high[i] >= current_sl
                    tp_hit = fixed_rr and low[i] <= take_profit
                    intrabar_ambiguous_exit = bool(stop_hit and tp_hit)
                    if intrabar_ambiguous_exit:
                        ambiguous_intrabar_exits += 1
                    if intrabar_policy == "optimistic" and intrabar_ambiguous_exit:
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif stop_hit:
                        exit_price = current_sl
                        exit_reason = "stop_loss" if current_sl == initial_sl else "trailing_stop"
                    elif tp_hit:
                        exit_price = take_profit
                        exit_reason = "take_profit"

            if exit_price is not None:
                gross = (
                    (exit_price - entry_price) * position_size
                    if position == POSITION_LONG
                    else (entry_price - exit_price) * position_size
                )
                held_exit = max(1, i - entry_idx + 1)
                funding = funding_cost_between(entry_idx, i, entry_price * position_size, direction)
                net = gross - (entry_price * position_size + exit_price * position_size) * (
                    commission_rate + slippage_rate
                ) - funding
                total_funding += funding
                balance += net
                equity_curve.append(balance)
                peak_equity = max(peak_equity, balance)
                dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0.0
                max_dd = max(max_dd, dd)
                gross_pnls.append(gross)
                net_pnls.append(net)
                if net > 0:
                    wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    if consecutive_losses >= int(params.get("max_consecutive_losses", 4)):
                        cooldown_until = i + int(params.get("cooldown_period_bars", 24))
                if float(params.get("drawdown_guard_pct", 0.0)) > 0.0 and dd * 100.0 >= float(
                    params.get("drawdown_guard_pct", 0.0)
                ):
                    dd_guard_until = i + int(params.get("drawdown_guard_cooldown_bars", 0))
                if balance <= 0:
                    break
                if include_trades:
                    trades_detail.append(
                        {
                            "entry_signal_index": entry_signal_idx,
                            "entry_index": entry_idx,
                            "exit_index": i,
                            "entry_signal_time": df.index[entry_signal_idx].isoformat()
                            if entry_signal_idx >= 0
                            else None,
                            "entry_time": df.index[entry_idx].isoformat(),
                            "exit_time": df.index[i].isoformat(),
                            "side": "long" if position == POSITION_LONG else "short",
                            "entry_price": round(float(entry_price), 8),
                            "exit_price": round(float(exit_price), 8),
                            "initial_stop_loss": round(float(initial_sl), 8),
                            "take_profit": round(float(take_profit), 8),
                            "position_size": round(float(position_size), 8),
                            "bars_held": int(held_exit),
                            "exit_reason": exit_reason,
                            "intrabar_policy": intrabar_policy,
                            "intrabar_ambiguous_exit": bool(intrabar_ambiguous_exit),
                            "gross_pnl": round(float(gross), 8),
                            "net_pnl": round(float(net), 8),
                            "funding": round(float(funding), 8),
                            "balance_after": round(float(balance), 8),
                            **entry_features,
                        }
                    )
                position = POSITION_NONE

        if position != POSITION_NONE:
            continue
        equity_curve.append(balance)
        if cooldown_until >= 0:
            if i >= cooldown_until:
                cooldown_until = -1
                consecutive_losses = 0
            else:
                continue
        if i < dd_guard_until:
            continue
        session = params.get("entry_session_mode", "all")
        if session == "eu_us" and not (13 <= int(hour_of_day[i]) <= 21):
            continue
        if session == "weekday_all" and int(day_of_week[i]) >= 5:
            continue
        if session == "weekday_eu_us" and (
            int(day_of_week[i]) >= 5 or not (13 <= int(hour_of_day[i]) <= 21)
        ):
            continue
        if session == "active_07_21" and not (7 <= int(hour_of_day[i]) <= 21):
            continue
        if session == "exclude_00_06" and int(hour_of_day[i]) <= 6:
            continue
        if session == "no_monday" and int(day_of_week[i]) == 0:
            continue
        if session == "no_tuesday" and int(day_of_week[i]) == 1:
            continue
        if session == "no_tue_fri" and int(day_of_week[i]) in {1, 4}:
            continue
        if session == "no_tue_fri_exclude_00_06" and (
            int(day_of_week[i]) in {1, 4} or int(hour_of_day[i]) <= 6
        ):
            continue
        if session == "no_mon_wed_sat" and int(day_of_week[i]) in {0, 2, 5}:
            continue
        if session == "exclude_08" and int(hour_of_day[i]) == 8:
            continue
        if session == "active_09_22" and not (9 <= int(hour_of_day[i]) <= 22):
            continue
        if session == "exclude_08_16_17" and int(hour_of_day[i]) in {8, 16, 17}:
            continue

        if bool(params.get("use_regime_filter", False)):
            adx = ind["adx"][i]
            atr = ind["atr_sl"][i]
            if not (np.isnan(adx) or np.isnan(atr)):
                atr_pct = (atr / close[i]) * 100.0 if close[i] > 0 else 0.0
                if not (
                    adx >= float(params.get("adx_threshold_regime", 0.0))
                    or atr_pct >= float(params.get("atr_percent_threshold_regime", 0.0))
                ):
                    continue

        ema_short = ind["ema_short"][i]
        ema_short_prev = ind["ema_short"][i - 1]
        ema_long = ind["ema_long"][i]
        ema_long_prev = ind["ema_long"][i - 1]
        if any(np.isnan(v) for v in [ema_short, ema_short_prev, ema_long, ema_long_prev]):
            continue
        entry_mode = params.get("entry_signal_mode", "crossover")
        long_signal = ema_short_prev < ema_long_prev and ema_short > ema_long
        if entry_mode == "trend_breakout":
            lookback = int(params.get("price_breakdown_period", 3))
            long_signal = False
            if i >= lookback:
                prior_high = float(np.max(high[i - lookback : i]))
                long_signal = bool(ema_short > ema_long and close[i] > prior_high)
        elif entry_mode == "donchian_breakout":
            lookback = int(params.get("price_breakdown_period", 3))
            long_signal = False
            if i > lookback:
                prior_high = float(np.max(high[i - lookback : i]))
                prior_high_before_prev = float(np.max(high[i - lookback - 1 : i - 1]))
                long_signal = bool(
                    ema_short > ema_long
                    and close[i] > prior_high
                    and close[i - 1] <= prior_high_before_prev
                )
        elif entry_mode == "cross_or_pullback":
            long_signal = bool(
                long_signal
                or (
                    ema_short > ema_long
                    and close[i - 1] <= ema_short_prev
                    and close[i] > ema_short
                )
            )
        short_signal = False
        if bool(params.get("allow_short_entries", True)):
            if entry_mode == "trend_breakout":
                lookback = int(params.get("price_breakdown_period", 3))
                short_signal = False
                if i >= lookback:
                    prior_low = float(np.min(low[i - lookback : i]))
                    short_signal = bool(ema_short < ema_long and close[i] < prior_low)
            elif entry_mode == "donchian_breakout":
                lookback = int(params.get("price_breakdown_period", 3))
                short_signal = False
                if i > lookback:
                    prior_low = float(np.min(low[i - lookback : i]))
                    prior_low_before_prev = float(np.min(low[i - lookback - 1 : i - 1]))
                    short_signal = bool(
                        ema_short < ema_long
                        and close[i] < prior_low
                        and close[i - 1] >= prior_low_before_prev
                    )
            else:
                short_signal = (
                    ema_short_prev > ema_long_prev
                    and ema_short < ema_long
                    and _short_entry_ok(i, close, low, ind, params)
                )
            if entry_mode == "cross_or_pullback":
                short_signal = bool(
                    short_signal
                    or (
                        ema_short < ema_long
                        and close[i - 1] >= ema_short_prev
                        and close[i] < ema_short
                    )
                )
        raw_long_signal = bool(long_signal)
        raw_short_signal = bool(short_signal)
        rejection_reasons: list[str] = []

        if long_signal or short_signal:
            if bool(params.get("use_funding_rate_filter", False)):
                recent_funding = _recent_funding_rate(
                    i,
                    funding_cumulative=funding_cumulative,
                    funding_rate_per_8h=funding_rate_per_8h,
                    lookback_hours=int(params.get("funding_rate_lookback_hours", 8)),
                )
                if long_signal and recent_funding > float(params.get("funding_rate_long_max", 1.0)):
                    long_signal = False
                    rejection_reasons.append("funding_long_above_max")
                if short_signal and recent_funding < float(params.get("funding_rate_short_min", -1.0)):
                    short_signal = False
                    rejection_reasons.append("funding_short_below_min")

            spread_min_atr = float(params.get("ema_spread_min_atr", 0.0))
            if spread_min_atr > 0.0:
                atr_for_spread = ind["atr_sl"][i]
                if np.isnan(atr_for_spread) or atr_for_spread <= 0:
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("ema_spread_atr_unavailable")
                else:
                    spread_atr = abs(ema_short - ema_long) / atr_for_spread
                    if spread_atr < spread_min_atr:
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("ema_spread_below_min_atr")

            if (long_signal or short_signal) and bool(params.get("use_h1_slope_filter", False)):
                lookback = int(params.get("h1_slope_lookback_bars", 24))
                lookback_idx = i - lookback
                if lookback_idx < 0:
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("h1_slope_lookback_unavailable")
                else:
                    ema_long_prev_slope = ind["ema_long"][lookback_idx]
                    if np.isnan(ema_long_prev_slope) or ema_long_prev_slope <= 0:
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("h1_slope_reference_unavailable")
                    else:
                        h1_slope_pct = (ema_long - ema_long_prev_slope) / ema_long_prev_slope
                        min_slope = float(params.get("h1_min_slope_pct", 0.0))
                        if long_signal and h1_slope_pct < min_slope:
                            long_signal = False
                            rejection_reasons.append("h1_long_slope_below_min")
                        if short_signal and h1_slope_pct > -min_slope:
                            short_signal = False
                            rejection_reasons.append("h1_short_slope_above_negative_min")

            if (long_signal or short_signal) and bool(params.get("use_h1_atr_percent_filter", False)):
                atr_for_filter = ind["atr_sl"][i]
                if np.isnan(atr_for_filter) or close[i] <= 0:
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("h1_atr_percent_unavailable")
                else:
                    h1_atr_pct = (atr_for_filter / close[i]) * 100.0
                    if h1_atr_pct < float(params.get("h1_atr_percent_min", 0.0)):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("h1_atr_percent_below_min")
                    elif (
                        float(params.get("h1_atr_percent_max", 100.0)) > 0.0
                        and h1_atr_pct > float(params.get("h1_atr_percent_max", 100.0))
                    ):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("h1_atr_percent_above_max")

            if bool(params.get("use_htf_ema_filter", False)):
                h4_ema = ind["h4_ema"][i]
                if not np.isnan(h4_ema):
                    if (long_signal and close[i] < h4_ema) or (short_signal and close[i] > h4_ema):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("htf_ema_against_direction")
            if (long_signal or short_signal) and bool(params.get("use_htf_regime_filter", False)):
                h4_adx = ind["h4_adx"][i]
                h4_atr = ind["h4_atr"][i]
                if np.isnan(h4_adx) or np.isnan(h4_atr) or close[i] <= 0:
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("htf_regime_reference_unavailable")
                else:
                    h4_atr_pct = (h4_atr / close[i]) * 100.0
                    if h4_adx < float(params.get("htf_adx_threshold", 0.0)):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("htf_adx_below_threshold")
                    elif h4_atr_pct < float(params.get("htf_atr_percent_min", 0.0)):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("htf_atr_percent_below_min")
                    elif (
                        float(params.get("htf_atr_percent_max", 100.0)) > 0.0
                        and h4_atr_pct > float(params.get("htf_atr_percent_max", 100.0))
                    ):
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("htf_atr_percent_above_max")
            if (long_signal or short_signal) and bool(params.get("use_htf_slope_filter", False)):
                lookback = int(params.get("htf_slope_lookback_bars", 24))
                lookback_idx = i - lookback
                if lookback_idx < 0:
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("htf_slope_lookback_unavailable")
                else:
                    h4_curr = ind["h4_ema"][i]
                    h4_prev = ind["h4_ema"][lookback_idx]
                    if np.isnan(h4_curr) or np.isnan(h4_prev) or h4_prev <= 0:
                        long_signal = False
                        short_signal = False
                        rejection_reasons.append("htf_slope_reference_unavailable")
                    else:
                        slope_pct = (h4_curr - h4_prev) / h4_prev
                        min_slope = float(params.get("htf_min_slope_pct", 0.0))
                        buffer_atr = float(params.get("htf_price_buffer_atr", 0.0))
                        block_min = float(params.get("htf_block_slope_min_pct", 0.0))
                        block_max = float(params.get("htf_block_slope_max_pct", 0.0))
                        atr_for_buffer = ind["atr_sl"][i]
                        if buffer_atr > 0.0 and (np.isnan(atr_for_buffer) or atr_for_buffer <= 0):
                            long_signal = False
                            short_signal = False
                            rejection_reasons.append("htf_price_buffer_atr_unavailable")
                        if (
                            bool(params.get("use_htf_slope_block_filter", False))
                            and block_max > block_min
                        ):
                            if long_signal and block_min < slope_pct <= block_max:
                                long_signal = False
                                rejection_reasons.append("htf_long_slope_blocked")
                            if short_signal and -block_max <= slope_pct < -block_min:
                                short_signal = False
                                rejection_reasons.append("htf_short_slope_blocked")
                        if long_signal:
                            if slope_pct < min_slope:
                                long_signal = False
                                rejection_reasons.append("htf_long_slope_below_min")
                            elif buffer_atr > 0.0 and close[i] < h4_curr + atr_for_buffer * buffer_atr:
                                long_signal = False
                                rejection_reasons.append("htf_long_price_inside_buffer")
                        if short_signal:
                            if slope_pct > -min_slope:
                                short_signal = False
                                rejection_reasons.append("htf_short_slope_above_negative_min")
                            elif buffer_atr > 0.0 and close[i] > h4_curr - atr_for_buffer * buffer_atr:
                                short_signal = False
                                rejection_reasons.append("htf_short_price_inside_buffer")
            if bool(params.get("use_adx_filter", False)) and ind["adx"][i] < float(params.get("adx_threshold", 25.0)):
                long_signal = False
                short_signal = False
                rejection_reasons.append("adx_below_threshold")
            if bool(params.get("use_volume_filter", False)) and volume[i] <= ind["vol_sma"][i]:
                long_signal = False
                short_signal = False
                rejection_reasons.append("volume_below_sma")
            if bool(params.get("use_rsi_filter", False)):
                rsi = ind["rsi"][i]
                if (long_signal and rsi < float(params.get("rsi_threshold_long", 50.0))) or (
                    short_signal and rsi > float(params.get("rsi_threshold_short", 45.0))
                ):
                    long_signal = False
                    short_signal = False
                    rejection_reasons.append("rsi_filter_rejected")

        signal = POSITION_LONG if long_signal else (POSITION_SHORT if short_signal else POSITION_NONE)
        if signal == POSITION_NONE:
            if include_entry_events and (raw_long_signal or raw_short_signal):
                entry_events.append(
                    {
                        "status": "rejected",
                        "entry_signal_index": i,
                        "entry_signal_time": df.index[i].isoformat(),
                        "raw_side": "long" if raw_long_signal else "short",
                        "skip_reasons": rejection_reasons or ["entry_filters_removed_signal"],
                        **_entry_feature_snapshot(
                            i,
                            close=close,
                            ind=ind,
                            params=params,
                            hour_of_day=hour_of_day,
                            day_of_week=day_of_week,
                            ema_short=ema_short,
                            ema_long=ema_long,
                        ),
                    }
                )
            continue
        atr_sl = ind["atr_sl"][i]
        if np.isnan(atr_sl) or atr_sl <= 0:
            if include_entry_events:
                entry_events.append(
                    {
                        "status": "rejected",
                        "entry_signal_index": i,
                        "entry_signal_time": df.index[i].isoformat(),
                        "raw_side": "long" if signal == POSITION_LONG else "short",
                        "skip_reasons": ["atr_sl_unavailable"],
                        **_entry_feature_snapshot(
                            i,
                            close=close,
                            ind=ind,
                            params=params,
                            hour_of_day=hour_of_day,
                            day_of_week=day_of_week,
                            ema_short=ema_short,
                            ema_long=ema_long,
                        ),
                    }
                )
            continue
        fill_idx = i + entry_delay_bars
        if fill_idx >= len(df):
            if include_entry_events:
                entry_events.append(
                    {
                        "status": "pending_next_bar_open",
                        "entry_signal_index": i,
                        "entry_signal_time": df.index[i].isoformat(),
                        "raw_side": "long" if signal == POSITION_LONG else "short",
                        "planned_entry_time": (
                            df.index[i] + (df.index[i] - df.index[i - 1])
                            if i > 0
                            else df.index[i] + pd.Timedelta(hours=entry_delay_bars)
                        ).isoformat(),
                        "skip_reasons": ["next_bar_open_unavailable"],
                        **_entry_feature_snapshot(
                            i,
                            close=close,
                            ind=ind,
                            params=params,
                            hour_of_day=hour_of_day,
                            day_of_week=day_of_week,
                            ema_short=ema_short,
                            ema_long=ema_long,
                        ),
                    }
                )
            continue
        fill_price = close[i] if entry_delay_bars <= 0 else open_[fill_idx]
        sl_distance = atr_sl * float(params.get("atr_multiplier_sl", 2.6))
        volatility_size_mult = _volatility_target_multiplier(i, close, params)
        drawdown_size_mult = _drawdown_sizing_multiplier(balance, peak_equity, params)
        adjusted_risk_pct = (
            float(params.get("risk_per_trade_percentage", 0.02))
            * volatility_size_mult
            * drawdown_size_mult
        )
        qty_raw = balance * adjusted_risk_pct / sl_distance
        qty = math.floor(qty_raw * (10**quantity_precision)) / (10**quantity_precision)
        if qty < min_trade_size:
            if include_entry_events:
                entry_events.append(
                    {
                        "status": "rejected",
                        "entry_signal_index": i,
                        "entry_signal_time": df.index[i].isoformat(),
                        "raw_side": "long" if signal == POSITION_LONG else "short",
                        "skip_reasons": ["quantity_below_min_trade_size"],
                        "quantity_raw": round(float(qty_raw), 8),
                        "quantity_rounded": round(float(qty), 8),
                        "volatility_size_multiplier": round(float(volatility_size_mult), 8),
                        **_entry_feature_snapshot(
                            i,
                            close=close,
                            ind=ind,
                            params=params,
                            hour_of_day=hour_of_day,
                            day_of_week=day_of_week,
                            ema_short=ema_short,
                            ema_long=ema_long,
                        ),
                    }
                )
            continue
        entry_features = _entry_feature_snapshot(
            i,
            close=close,
            ind=ind,
            params=params,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            ema_short=ema_short,
            ema_long=ema_long,
        )
        entry_features["volatility_size_multiplier"] = round(float(volatility_size_mult), 8)
        position = signal
        entry_price = fill_price
        position_size = qty
        entry_idx = fill_idx
        entry_signal_idx = i
        if signal == POSITION_LONG:
            initial_sl = fill_price - sl_distance
            take_profit = fill_price + sl_distance * float(params.get("risk_reward_ratio", 3.0))
        else:
            initial_sl = fill_price + sl_distance
            take_profit = fill_price - sl_distance * float(params.get("risk_reward_ratio", 3.0))
        current_sl = initial_sl
        if include_entry_events:
            entry_events.append(
                {
                    "status": "accepted",
                    "entry_signal_index": i,
                    "entry_index": entry_idx,
                    "entry_signal_time": df.index[i].isoformat(),
                    "entry_time": df.index[entry_idx].isoformat(),
                    "side": "long" if signal == POSITION_LONG else "short",
                    "theoretical_entry_price": round(float(entry_price), 8),
                    "initial_stop_loss": round(float(initial_sl), 8),
                    "take_profit": round(float(take_profit), 8),
                    "position_size": round(float(position_size), 8),
                    "skip_reasons": [],
                    **entry_features,
                }
            )

    trades = len(net_pnls)
    result = {
        "final_balance": round(balance, 2),
        "total_net_pnl": round(balance - initial_balance, 2),
        "total_net_pnl_percentage": round((balance - initial_balance) / initial_balance * 100.0, 2),
        "num_trades": trades,
        "num_wins": wins,
        "num_losses": trades - wins,
        "win_rate_percentage": round((wins / trades) * 100.0 if trades else 0.0, 2),
        "profit_factor": round(_profit_factor(net_pnls), 2) if math.isfinite(_profit_factor(net_pnls)) else "inf",
        "net_profit_factor": round(_profit_factor(net_pnls), 2) if math.isfinite(_profit_factor(net_pnls)) else "inf",
        "gross_profit_factor": round(_profit_factor(gross_pnls), 2) if math.isfinite(_profit_factor(gross_pnls)) else "inf",
        "max_drawdown_percentage": round(max_dd * 100.0, 2),
        "intrabar_policy": intrabar_policy,
        "ambiguous_intrabar_exits": int(ambiguous_intrabar_exits),
        "funding_model": "actual_funding_events" if funding_cumulative is not None else "constant_per_8h",
        "total_funding": round(float(total_funding), 8),
    }
    if include_equity and include_trades and include_entry_events:
        return result, np.array(equity_curve, dtype=np.float64), trades_detail, entry_events
    if include_equity and include_trades:
        return result, np.array(equity_curve, dtype=np.float64), trades_detail
    if include_trades and include_entry_events:
        return result, trades_detail, entry_events
    if include_equity:
        return result, np.array(equity_curve, dtype=np.float64)
    if include_trades:
        return result, trades_detail
    if include_entry_events:
        return result, entry_events
    return result


def _load_cuda_row(path: Path, rank: int) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    for row in obj.get("results", []):
        if int(row["rank"]) == rank:
            return obj, row
    raise SystemExit(f"No rank {rank} in {path}")


def _diff(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_net_pnl_percentage",
        "num_trades",
        "win_rate_percentage",
        "net_profit_factor",
        "gross_profit_factor",
        "max_drawdown_percentage",
    ]
    out = {}
    for key in keys:
        c = cpu.get(key)
        g = gpu.get(key, gpu.get("profit_factor") if key == "net_profit_factor" else None)
        if isinstance(c, str) or isinstance(g, str):
            out[key] = {"cpu": c, "gpu": g, "abs_diff": None}
        else:
            out[key] = {"cpu": c, "gpu": g, "abs_diff": round(abs(_num(c) - _num(g)), 6)}
    return out


def _diff_pass(diff: dict[str, Any], pnl_tol: float, pf_tol: float, mdd_tol: float, trade_tol: int) -> bool:
    return (
        _num(diff["total_net_pnl_percentage"]["abs_diff"]) <= pnl_tol
        and _num(diff["num_trades"]["abs_diff"]) <= trade_tol
        and _num(diff["net_profit_factor"]["abs_diff"]) <= pf_tol
        and _num(diff["gross_profit_factor"]["abs_diff"]) <= pf_tol
        and _num(diff["max_drawdown_percentage"]["abs_diff"]) <= mdd_tol
    )


def _policy_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": _num(metrics.get("total_net_pnl_percentage")),
        "net_pf": _num(metrics.get("net_profit_factor", metrics.get("profit_factor"))),
        "gross_pf": _num(metrics.get("gross_profit_factor")),
        "mdd_pct": _num(metrics.get("max_drawdown_percentage")),
        "trades": int(metrics.get("num_trades", 0)),
        "win_rate_pct": _num(metrics.get("win_rate_percentage")),
        "ambiguous_intrabar_exits": int(metrics.get("ambiguous_intrabar_exits", 0)),
    }


def _intrabar_policy_comparison(conservative: dict[str, Any], optimistic: dict[str, Any]) -> dict[str, Any]:
    conservative_summary = _policy_summary(conservative)
    optimistic_summary = _policy_summary(optimistic)
    return {
        "conservative": conservative_summary,
        "optimistic": optimistic_summary,
        "delta_optimistic_minus_conservative": {
            "return_pct": round(optimistic_summary["return_pct"] - conservative_summary["return_pct"], 4),
            "net_pf": round(optimistic_summary["net_pf"] - conservative_summary["net_pf"], 4),
            "gross_pf": round(optimistic_summary["gross_pf"] - conservative_summary["gross_pf"], 4),
            "mdd_pct": round(optimistic_summary["mdd_pct"] - conservative_summary["mdd_pct"], 4),
            "trades": optimistic_summary["trades"] - conservative_summary["trades"],
        },
        "note": "CUDA search and strict gates use conservative SL-first behavior; optimistic TP-first is reported as an OHLC intrabar ambiguity band.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff a CUDA top-result row against a CPU reference implementation.")
    parser.add_argument("--cuda-json", required=True, type=Path)
    parser.add_argument("--csv", default="data/BTCUSDT_1h.csv", type=Path)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pnl-tol", type=float, default=0.01)
    parser.add_argument("--pf-tol", type=float, default=0.01)
    parser.add_argument("--mdd-tol", type=float, default=0.01)
    parser.add_argument("--trade-tol", type=int, default=0)
    parser.add_argument("--include-intrabar-comparison", action="store_true")
    parser.add_argument("--funding-csv", type=Path, default=None, help="Override or provide actual funding-rate CSV for CPU reference diff.")
    args = parser.parse_args()

    obj, row = _load_cuda_row(args.cuda_json, args.rank)
    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    df = _load_ohlcv(csv_path, obj["period_start"], obj["period_end"])
    funding_csv = args.funding_csv
    if funding_csv is None and obj.get("funding_model") == "actual_funding_events" and obj.get("funding_rate_csv"):
        funding_csv = Path(obj["funding_rate_csv"])
    funding_cumulative = load_funding_cumulative(df, funding_csv) if funding_csv is not None else None
    cpu = cpu_reference_backtest(
        df,
        row["parameters"],
        initial_balance=float(row["performance"].get("initial_balance", 10_000.0)),
        commission_rate=float(obj.get("commission_rate", 0.0005)),
        slippage_rate=float(obj.get("slippage_rate", 0.0002)),
        entry_delay_bars=int(obj.get("entry_delay_bars", 1)),
        funding_rate_per_8h=float(obj.get("funding_rate_per_8h", 0.0)),
        funding_cumulative=funding_cumulative,
    )
    diff = _diff(cpu, row["performance"])
    payload = {
        "cuda_json": str(args.cuda_json),
        "rank": args.rank,
        "period_start": obj["period_start"],
        "period_end": obj["period_end"],
        "funding_model": "actual_funding_events" if funding_cumulative is not None else "constant_per_8h",
        "funding_csv": str(funding_csv) if funding_csv is not None else None,
        "param_id": row["performance"]["param_id"],
        "cpu_reference": cpu,
        "cuda_performance": row["performance"],
        "diff": diff,
        "diff_pass": _diff_pass(diff, args.pnl_tol, args.pf_tol, args.mdd_tol, args.trade_tol),
        "tolerances": {
            "pnl_pct": args.pnl_tol,
            "pf": args.pf_tol,
            "mdd_pct": args.mdd_tol,
            "trades": args.trade_tol,
        },
    }
    if args.include_intrabar_comparison:
        optimistic = cpu_reference_backtest(
            df,
            row["parameters"],
            initial_balance=float(row["performance"].get("initial_balance", 10_000.0)),
            commission_rate=float(obj.get("commission_rate", 0.0005)),
            slippage_rate=float(obj.get("slippage_rate", 0.0002)),
            entry_delay_bars=int(obj.get("entry_delay_bars", 1)),
            funding_rate_per_8h=float(obj.get("funding_rate_per_8h", 0.0)),
            intrabar_policy="optimistic",
            funding_cumulative=funding_cumulative,
        )
        payload["intrabar_policy_comparison"] = _intrabar_policy_comparison(cpu, optimistic)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = args.out if args.out.is_absolute() else Path.cwd() / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
