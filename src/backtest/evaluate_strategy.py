import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import components from src
from src.filters.regime_filter import passes_regime_filter, compute_adx
from src.filters.volatility_filter import passes_vol_filter, compute_atr
from src.exits.trailing import ATRTrailingStop
from src.position.sizing import RiskSizer
from src.policy.time_windows import hour_allowed, weekday_allowed

# --- Configuration (adjust as needed) ---
# This path should point to your pre-generated dollar bars
DOLLAR_BARS_PATH = "C:\\Monilusion\\data\\dollar_bars_BTCUSDT_2021-2025.parquet"

def evaluate_strategy(params: dict) -> dict:
    """
    Evaluates the trading strategy with the given parameters.
    This is a basic placeholder for demonstration.
    A full backtesting engine would be more complex.

    Args:
        params (dict): A dictionary of strategy parameters.

    Returns:
        dict: A dictionary of backtest results (e.g., profit_factor, win_rate, max_dd).
    """
    print(f"Evaluating strategy with parameters: {params}")

    # --- 1. Load Data ---
    try:
        df = pd.read_parquet(DOLLAR_BARS_PATH)
        # Check if 'open_time' is a column or already the index
        if 'open_time' in df.columns:
            df = df.set_index('open_time').sort_index()
        else: # Assume it's already indexed by open_time
            df.index.name = 'open_time'
            df = df.sort_index()

        # Ensure necessary columns are present and numeric
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Missing required column '{col}' in dollar bars.")
                return {"profit_factor": 0, "win_rate": 0, "max_dd": 0}
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=required_cols, inplace=True)

        if df.empty:
            print("Error: Loaded DataFrame is empty after cleaning.")
            return {"profit_factor": 0, "win_rate": 0, "max_dd": 0}

        # Pre-compute ATR and ADX for efficiency
        df['atr'] = compute_atr(df, period=params['volatility']['atr_period'])
        adx_df = compute_adx(df, period=params['volatility']['atr_period']) # Corrected: use volatility atr_period
        df['adx'] = adx_df['adx']
        df['plus_di'] = adx_df['plus_di']
        df['minus_di'] = adx_df['minus_di']

        df.dropna(inplace=True) # Drop NaNs introduced by indicator calculations

    except FileNotFoundError:
        print(f"Error: Dollar bars file not found at {DOLLAR_BARS_PATH}")
        return {"profit_factor": 0, "win_rate": 0, "max_dd": 0}
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        return {"profit_factor": 0, "win_rate": 0, "max_dd": 0}

    # --- 2. Initialize Backtest State ---
    trades = []
    current_position = None # {'side': 'Long'/'Short', 'entry_price': float, 'entry_time': datetime, 'atr_stop_instance': ATRTrailingStop}
    risk_sizer = RiskSizer(
        loss_streak_k=params['sizing']['loss_streak_k'],
        cooldown_factor=params['sizing']['cooldown_factor'],
        day_loss_cap_r=params['sizing']['day_loss_cap_r'],
        week_loss_cap_r=params['sizing']['week_loss_cap_r']
    )
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    max_drawdown = 0.0
    peak_equity = 0.0
    initial_capital = 10000 # Example initial capital

    # --- 3. Backtest Loop ---
    for i in range(len(df)):
        current_bar = df.iloc[i]
        current_time = current_bar.name # The index is open_time
        current_price = current_bar['close']
        current_atr = current_bar['atr']

        # --- Apply Time Filters ---
        if not hour_allowed(current_time, params['time']['allow_hours'], params['time']['deny_hours']):
            if current_position: # If position open, check if it's time to exit due to time filter
                # For simplicity, we'll assume time filters only prevent new entries
                # A real backtest would check if an open position needs to be closed due to time.
                pass
            continue # Skip if current hour is not allowed

        if not weekday_allowed(current_time, params['time']['allow_days'], params['time']['deny_days']):
            if current_position:
                pass
            continue # Skip if current weekday is not allowed

        # --- Manage Open Position ---
        if current_position:
            elapsed_min = (current_time - current_position['entry_time']).total_seconds() / 60
            action, stop_level = current_position['atr_stop_instance'].update(
                current_price=current_price,
                entry_price=current_position['entry_price'],
                side=current_position['side'],
                elapsed_min=elapsed_min,
                atr=current_atr
            )

            if action == "exit":
                # Close position
                exit_price = current_price
                pnl = (exit_price - current_position['entry_price']) if current_position['side'] == 'Long' else (current_position['entry_price'] - exit_price)
                
                # For simplicity, assume 1 unit of trade, PnL is directly price difference
                # In a real system, this would be PnL in currency based on position size
                
                trades.append({
                    'entry_time': current_position['entry_time'],
                    'exit_time': current_time,
                    'side': current_position['side'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                # Update RiskSizer (assuming PnL in R units, here just using raw PnL for simplicity)
                # A real system would convert PnL to R units based on base_r
                risk_sizer.update_after_trade(pnl_r=pnl, ts=current_time) 
                
                current_position = None # Close position

        # --- Check for New Entry (only if no position is open) ---
        if not current_position:
            # Apply Risk Sizer
            size_multiplier = risk_sizer.size(current_time)
            if size_multiplier == 0.0:
                continue # Do not trade due to risk management

            # Apply Regime Filter
            # Note: passes_regime_filter expects df, idx, adx_min, ema_fast, ema_slow, side
            # We need to pass the current bar's data for ADX and EMAs
            # For simplicity, we'll use the pre-computed ADX and assume EMAs are also pre-computed or calculated on the fly
            # The current implementation of passes_regime_filter recomputes EMAs, which is fine for a placeholder.
            
            # Check for Long entry
            if passes_regime_filter(
                df.loc[:current_time], # Pass historical data up to current time
                current_time,
                adx_min=params['regime']['adx_min'],
                ema_fast=params['regime']['ema_fast'],
                ema_slow=params['regime']['ema_slow'],
                side="Long",
                price_col="close"
            ) and passes_vol_filter(
                df.loc[:current_time],
                current_time,
                min_move_pct=params['volatility']['min_move_pct'],
                atr_period=params['volatility']['atr_period'],
                price_col="close"
            ):
                # Enter Long position
                current_position = {
                    'side': 'Long',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'atr_stop_instance': ATRTrailingStop(
                        atr_mult=params['exits']['atr_mult'],
                        time_stop_min=params['exits']['time_stop_min'],
                        max_hold_min=params['exits']['max_hold_min']
                    )
                }
                # print(f"Opened Long at {current_price} on {current_time}")

            # Check for Short entry
            elif passes_regime_filter(
                df.loc[:current_time],
                current_time,
                adx_min=params['regime']['adx_min'],
                ema_fast=params['regime']['ema_fast'],
                ema_slow=params['regime']['ema_slow'],
                side="Short",
                price_col="close"
            ) and passes_vol_filter(
                df.loc[:current_time],
                current_time,
                min_move_pct=params['volatility']['min_move_pct'],
                atr_period=params['volatility']['atr_period'],
                price_col="close"
            ):
                # Enter Short position
                current_position = {
                    'side': 'Short',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'atr_stop_instance': ATRTrailingStop(
                        atr_mult=params['exits']['atr_mult'],
                        time_stop_min=params['exits']['time_stop_min'],
                        max_hold_min=params['exits']['max_hold_min']
                    )
                }
                # print(f"Opened Short at {current_price} on {current_time}")

        # --- Update Equity and Max Drawdown ---
        current_equity = initial_capital + total_pnl
        peak_equity = max(peak_equity, current_equity)
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    # --- 4. Calculate Final Metrics ---
    total_trades = len(trades)
    profit_factor = (sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                     (abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) + 1e-9)) if total_trades > 0 else 0
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    results = {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_dd": max_drawdown * 100 # Convert to percentage
    }
    return results

# Example usage (for testing this file directly)
if __name__ == "__main__":
    # Dummy parameters for testing
    dummy_params = {
        "time": {"allow_hours": [], "deny_hours": [], "allow_days": [], "deny_days": []},
        "volatility": {"atr_period": 14, "min_move_pct": 0.5},
        "regime": {"adx_min": 20.0, "ema_fast": 10, "ema_slow": 50, "atr_period": 14}, # Added atr_period for ADX
        "exits": {"atr_mult": 1.5, "time_stop_min": 720, "max_hold_min": 2880},
        "sizing": {"loss_streak_k": 3, "cooldown_factor": 0.5, "day_loss_cap_r": 2.0, "week_loss_cap_r": 5.0},
    }
    
    # Create a dummy dollar_bars_BTCUSDT_2021-2025.parquet if it doesn't exist for testing
    if not os.path.exists(DOLLAR_BARS_PATH):
        print(f"Creating dummy dollar bars for testing at {DOLLAR_BARS_PATH}...")
        os.makedirs(os.path.dirname(DOLLAR_BARS_PATH), exist_ok=True)
        dates = pd.to_datetime(pd.date_range('2021-01-01', '2021-01-05', freq='h')) # Changed 'H' to 'h'
        data = {
            'open': np.random.rand(len(dates)) * 1000 + 50000,
            'high': np.random.rand(len(dates)) * 1000 + 51000,
            'low': np.random.rand(len(dates)) * 1000 + 49000,
            'close': np.random.rand(len(dates)) * 1000 + 50000,
            'volume': np.random.rand(len(dates)) * 10000
        }
        dummy_df = pd.DataFrame(data, index=dates)
        dummy_df.index.name = 'open_time'
        dummy_df.to_parquet(DOLLAR_BARS_PATH)
        print("Dummy dollar bars created.")

    results = evaluate_strategy(dummy_params)
    print("\n--- Dummy Backtest Results ---")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Max Drawdown: {results['max_dd']:.2f}%")
