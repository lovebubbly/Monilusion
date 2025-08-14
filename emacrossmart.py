# -*- coding: utf-8 -*-
# EMA Crossover Strategy Backtester (V6.4 - Final Robustness Patch)
# pip install python-binance pandas numpy python-dotenv pandas_ta openpyxl numba cupy-cuda12x (or your cuda version)

import os
import time
from dotenv import load_dotenv
from binance.client import Client, BinanceAPIException
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import logging
from datetime import datetime, timedelta
import itertools
import heapq
from numba import cuda
import cupy as cp
import gc
import json

# --- 로깅 설정 ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(filename)s:%(lineno)d - %(message)s')
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)
logger.addHandler(log_handler_stream)

# --- .env 파일 로드 ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env 파일 로드 완료.")
else:
    logger.warning("경고: .env 파일 없음.")

# --- 오프라인 모드 설정 ---
OFFLINE_OHLCV_H1 = os.getenv('OFFLINE_OHLCV_H1')
OFFLINE_OHLCV_H4 = os.getenv('OFFLINE_OHLCV_H4')
OFFLINE_MODE = bool(OFFLINE_OHLCV_H1)

# --- API 키 설정 ---
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')
if not OFFLINE_MODE:
    if not api_key or not api_secret:
        logger.error("오류: API 키 필요! (.env에 BINANCE_API_KEY/BINANCE_SECRET_KEY) 또는 OFFLINE_OHLCV_H1 설정")
        exit()
    client = Client(api_key, api_secret)
    logger.info("바이낸스 클라이언트 초기화 완료 (백테스팅용).")
else:
    logger.info("오프라인 모드 활성화: 로컬 OHLCV 파일에서 데이터 로드")

# --- 백테스팅 공통 설정 ---
symbol_backtest = 'BTCUSDT'
interval_primary_bt = '1h'
interval_htf_bt = '4h'
num_klines_to_fetch = None
start_date_str = "2023-01-01"
end_date_str = "2025-05-01"
initial_balance = 10000
commission_rate_backtest = 0.0005
slippage_rate_per_trade = 0.0002
min_trade_size_btc = 0.001
price_precision_bt = 2
quantity_precision_bt = 3

# --- 테스트할 파라미터 범위 정의 (V6.3 - 패치 적용) ---
param_ranges = {
    # 모든 필터 자동 탐색
    'use_regime_filter': [True, False],
    'use_htf_ema_filter': [True, False],
    'use_adx_filter': [True, False],
    'use_volume_filter': [True, False],
    'use_rsi_filter': [True, False],

    # 주요 파라미터 범위
    'ema_short_h1': [10, 12, 14],
    'ema_long_h1': [50, 100],
    'ema_htf': [50, 100],
    'adx_period': [14],
    'adx_threshold': [25, 30],
    'atr_period_sl': [14, 21],
    'atr_multiplier_sl': [2.2, 2.6],
    'exit_strategy_type': ['TrailingATR', 'FixedRR'],
    'risk_reward_ratio': [2.0, 2.5, 3.0],
    'trailing_atr_period': [14, 21],
    'trailing_atr_multiplier': [1.5, 2.0],
    'volume_sma_period': [20, 30],
    'rsi_period': [21],
    'rsi_threshold_long': [50, 52],
    'rsi_threshold_short': [45, 47],
    'risk_per_trade_percentage': [0.01, 0.015, 0.02],
    
    # ✅ 패치 2: 레짐 게이트 현실화
    'adx_threshold_regime': [18.0, 22.0],
    'atr_percent_threshold_regime': [0.3, 0.6],
    
    'time_stop_period_hours': [24, 48],
    'profit_threshold_for_trail': [0.5, 1.0],
    'max_consecutive_losses': [3, 4],
    'cooldown_period_bars': [12, 24],
    'adx_threshold_for_short': [25.0, 30.0],
    'price_breakdown_period': [3, 5],
    'rsi_momentum_threshold': [40.0, 45.0],
}


# --- GPU 커널 및 백테스팅 관련 상수 ---
BATCH_SIZE = 1000000
POSITION_NONE_GPU, POSITION_LONG_GPU, POSITION_SHORT_GPU = 0, 1, -1
EXIT_REASON_NONE_GPU, EXIT_REASON_SL_GPU, EXIT_REASON_TP_GPU, EXIT_REASON_TRAIL_SL_GPU, EXIT_REASON_TIME_STOP_GPU = -1, 0, 1, 2, 3
EXIT_STRATEGY_FIXED_RR_GPU, EXIT_STRATEGY_TRAILING_ATR_GPU = 0, 1

# --- Helper Functions (데이터 로딩, JSON 직렬화 등) ---
def _read_ohlcv_local(path):
    try:
        if path.lower().endswith('.csv'): df = pd.read_csv(path)
        elif path.lower().endswith(('.parquet', '.pq')): df = pd.read_parquet(path)
        else: logger.error(f"지원하지 않는 파일 형식: {path}"); return None
    except Exception as e: logger.error(f"로컬 OHLCV 로딩 실패: {e}"); return None
    cols = {c.lower().strip(): c for c in df.columns}
    if 'Open time' not in df.columns and 'open time' not in cols:
        if df.index.name and 'time' in str(df.index.name).lower(): df = df.reset_index().rename(columns={df.columns[0]: 'Open time'})
        elif 'timestamp' in cols: df.rename(columns={cols['timestamp']: 'Open time'}, inplace=True)
        elif 'date' in cols: df.rename(columns={cols['date']: 'Open time'}, inplace=True)
    rename_map = {}
    for want in ['Open','High','Low','Close','Volume']:
        low = want.lower()
        if want not in df.columns and low in cols: rename_map[cols[low]] = want
    if rename_map: df.rename(columns=rename_map, inplace=True)
    required = ['Open time','Open','High','Low','Close','Volume']
    if not all(col in df.columns for col in required):
        logger.error(f"로컬 파일에 필요한 컬럼 없음. 필요: {required}, 현재: {list(df.columns)}"); return None
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df[required].copy()
    df.set_index('Open time', inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    logger.info(f"로컬 OHLCV 로드 완료: {path}, {df.index.min()} ~ {df.index.max()}, rows={len(df)}")
    return df

def get_historical_data(symbol, interval, start_str=None, end_str=None, limit=1000, max_klines=None):
    df = pd.DataFrame()
    logger.info(f"데이터 로딩 시작: Symbol={symbol}, Interval={interval}, Start={start_str}, End={end_str}, MaxKlines={max_klines}")
    start_time_ms_calc, end_time_ms_calc = None, None
    supported_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    def parse_datetime_str(date_str):
        if date_str is None: return None
        for fmt in supported_formats:
            try: return int(datetime.strptime(date_str, fmt).timestamp() * 1000)
            except ValueError: continue
        logger.error(f"시간 형식 오류: {date_str}. 지원 형식: {supported_formats}"); return None
    start_time_ms_calc = parse_datetime_str(start_str)
    end_time_ms_calc = parse_datetime_str(end_str)
    if start_time_ms_calc is None and max_klines:
        try:
            interval_td = pd.Timedelta(interval)
            start_time_dt = datetime.now() - (interval_td * max_klines)
            start_time_ms_calc = int(start_time_dt.timestamp() * 1000)
        except ValueError as e: logger.error(f"Interval 형식 오류 '{interval}'를 Timedelta로 변환 실패: {e}"); return None
    if end_time_ms_calc is None: end_time_ms_calc = int(datetime.now().timestamp() * 1000)
    if start_time_ms_calc is None: logger.error("시작 시간 또는 최대 Klines 수 지정 필요."); return None
    current_start_time_ms, total_fetched_klines, klines_list = start_time_ms_calc, 0, []
    while (not max_klines or total_fetched_klines < max_klines) and current_start_time_ms < end_time_ms_calc:
        fetch_limit = min(limit, max_klines - total_fetched_klines) if max_klines else limit
        if fetch_limit <= 0: break
        try:
            klines_batch = client.futures_klines(symbol=symbol, interval=interval, startTime=current_start_time_ms, limit=fetch_limit, endTime=end_time_ms_calc)
            if not klines_batch: break
            klines_list.extend(klines_batch)
            current_start_time_ms = klines_batch[-1][6] + 1
            total_fetched_klines += len(klines_batch)
        except BinanceAPIException as e: logger.error(f"API 오류 (데이터 로딩): {e}"); time.sleep(5)
        except Exception as e_gen: logger.error(f"알 수 없는 오류 (데이터 로딩): {e_gen}"); break
        time.sleep(0.1)
    if not klines_list: logger.error("데이터 가져오기 실패."); return None
    df = pd.DataFrame(klines_list, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
    df.drop_duplicates(inplace=True); df.sort_index(inplace=True)
    if max_klines and len(df) > max_klines: df = df.iloc[-max_klines:]
    logger.info(f"총 {len(df)}개 Klines 로드 완료. 기간: {df.index.min()} ~ {df.index.max()}")
    return df

def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, (set, tuple)): return list(obj)
    try:
        import cupy as _cp
        if isinstance(obj, _cp.ndarray): return _cp.asnumpy(obj).tolist()
    except ImportError: pass
    return str(obj)

def get_unique_indicator_params_and_map(param_ranges_dict):
    unique_params_values = {
        'ema_lengths': sorted(list(set(param_ranges_dict.get('ema_short_h1', []) + param_ranges_dict.get('ema_long_h1', []) + param_ranges_dict.get('ema_htf', [])))),
        'adx_periods': sorted(list(set(param_ranges_dict.get('adx_period', [])))),
        'atr_periods': sorted(list(set(param_ranges_dict.get('atr_period_sl', []) + param_ranges_dict.get('trailing_atr_period', [])))),
        'volume_sma_periods': sorted(list(set(param_ranges_dict.get('volume_sma_period', [])))),
        'rsi_periods': sorted(list(set(param_ranges_dict.get('rsi_period', [])))),
    }
    for key, default_val in {'ema_lengths': [20], 'adx_periods': [14], 'atr_periods': [14], 'volume_sma_periods': [20], 'rsi_periods': [14]}.items():
        if not unique_params_values[key]: unique_params_values[key] = default_val
    indicator_map, current_idx = {}, 0
    for length in unique_params_values['ema_lengths']: indicator_map[('ema', length)] = current_idx; current_idx += 1
    for period in unique_params_values['atr_periods']: indicator_map[('atr', period)] = current_idx; current_idx += 1
    for period in unique_params_values['adx_periods']:
        indicator_map[('adx', period)], indicator_map[('dmp', period)], indicator_map[('dmn', period)] = current_idx, current_idx + 1, current_idx + 2
        current_idx += 3
    for period in unique_params_values['volume_sma_periods']: indicator_map[('vol_sma', period)] = current_idx; current_idx += 1
    for period in unique_params_values['rsi_periods']: indicator_map[('rsi', period)] = current_idx; current_idx += 1
    logger.info(f"고유 지표 파라미터 추출 및 매핑 완료. 총 {current_idx}개의 고유 지표 시리즈 생성 예정.")
    return unique_params_values, indicator_map, current_idx

def precompute_all_indicators_for_gpu(df_ohlcv, unique_params_dict, interval_suffix, data_length):
    if df_ohlcv is None or df_ohlcv.empty: logger.warning(f"사전 계산을 위한 {interval_suffix} 데이터 없음."); return None
    _, temp_indicator_map, num_total_series = get_unique_indicator_params_and_map(param_ranges)
    master_indicator_array = np.full((num_total_series, data_length), np.nan, dtype=np.float64)
    logger.info(f"{interval_suffix} 데이터에 대한 마스터 지표 배열 생성 ({master_indicator_array.shape}).")
    for length in unique_params_dict.get('ema_lengths', []):
        if length > 0 and (idx := temp_indicator_map.get(('ema', length))) is not None:
            try:
                if (s := df_ohlcv.ta.ema(length=length, append=False)) is not None and len(s) == data_length: master_indicator_array[idx, :] = s.to_numpy()
            except Exception as e: logger.error(f"EMA_{length}_{interval_suffix} 계산 오류: {e}")
    for period in unique_params_dict.get('atr_periods', []):
        if period > 0 and (idx := temp_indicator_map.get(('atr', period))) is not None:
            try:
                if (s := df_ohlcv.ta.atr(length=period, append=False)) is not None and len(s) == data_length: master_indicator_array[idx, :] = s.to_numpy()
            except Exception as e: logger.error(f"ATR_{period}_{interval_suffix} 계산 오류: {e}")
    for period in unique_params_dict.get('adx_periods', []):
        if period > 0 and (adx_idx := temp_indicator_map.get(('adx', period))) is not None:
            try:
                if (adx_df := df_ohlcv.ta.adx(length=period, append=False)) is not None:
                    for col_name, map_key in [(f"ADX_{period}", 'adx'), (f"DMP_{period}", 'dmp'), (f"DMN_{period}", 'dmn')]:
                        if col_name in adx_df and (idx := temp_indicator_map.get((map_key, period))) is not None and len(adx_df[col_name]) == data_length:
                            master_indicator_array[idx, :] = adx_df[col_name].to_numpy()
            except Exception as e: logger.error(f"ADX/DMP/DMN_{period}_{interval_suffix} 계산 오류: {e}")
    if 'Volume' in df_ohlcv.columns:
        for period in unique_params_dict.get('volume_sma_periods', []):
            if period > 0 and (idx := temp_indicator_map.get(('vol_sma', period))) is not None:
                try:
                    if (s := ta.sma(df_ohlcv['Volume'], length=period, append=False)) is not None and len(s) == data_length: master_indicator_array[idx, :] = s.to_numpy()
                except Exception as e: logger.error(f"Volume SMA_{period}_{interval_suffix} 계산 오류: {e}")
    for period in unique_params_dict.get('rsi_periods', []):
        if period > 0 and (idx := temp_indicator_map.get(('rsi', period))) is not None:
            try:
                if (s := df_ohlcv.ta.rsi(length=period, append=False)) is not None and len(s) == data_length: master_indicator_array[idx, :] = s.to_numpy()
            except Exception as e: logger.error(f"RSI_{period}_{interval_suffix} 계산 오류: {e}")
    logger.info(f"{interval_suffix} 데이터에 대한 마스터 지표 배열 계산 완료.")
    return master_indicator_array

def iter_dependent_param_combos(pr):
    """의존성과 상호 배타성을 고려하여 유효한 파라미터 조합만 생성하는 제너레이터."""
    filter_options = {}
    for key, use_key in [('adx', 'use_adx_filter'), ('vol', 'use_volume_filter'), ('rsi', 'use_rsi_filter'), ('htf', 'use_htf_ema_filter'), ('regime', 'use_regime_filter')]:
        filter_options[key] = []
        if True in pr.get(use_key, [False]):
            if key == 'adx': combos = itertools.product(pr['adx_period'], pr['adx_threshold'])
            elif key == 'vol': combos = itertools.product(pr['volume_sma_period'])
            elif key == 'rsi': combos = itertools.product(pr['rsi_period'], pr['rsi_threshold_long'], pr['rsi_threshold_short'])
            elif key == 'htf': combos = itertools.product(pr['ema_htf'])
            elif key == 'regime': combos = itertools.product(pr['adx_threshold_regime'], pr['atr_percent_threshold_regime'])
            else: combos = [()]
            
            for c in combos:
                d = {use_key: True}
                if key == 'adx': d.update({'adx_period': c[0], 'adx_threshold': c[1]})
                elif key == 'vol': d.update({'volume_sma_period': c[0]})
                elif key == 'rsi': d.update({'rsi_period': c[0], 'rsi_threshold_long': c[1], 'rsi_threshold_short': c[2]})
                elif key == 'htf': d.update({'ema_htf': c[0]})
                elif key == 'regime': d.update({'adx_threshold_regime': c[0], 'atr_percent_threshold_regime': c[1]})
                filter_options[key].append(d)
        
        if False in pr.get(use_key, [False]):
            d = {use_key: False}
            if key == 'adx': d.update({'adx_period': 14, 'adx_threshold': 20})
            elif key == 'vol': d.update({'volume_sma_period': 20})
            elif key == 'rsi': d.update({'rsi_period': 14, 'rsi_threshold_long': 50, 'rsi_threshold_short': 50})
            elif key == 'htf': d.update({'ema_htf': 50})
            elif key == 'regime': d.update({'adx_threshold_regime': 0, 'atr_percent_threshold_regime': 0})
            filter_options[key].append(d)

    base_keys = [
        'ema_short_h1', 'ema_long_h1', 'atr_period_sl', 'atr_multiplier_sl',
        'risk_per_trade_percentage', 'time_stop_period_hours', 'profit_threshold_for_trail', 
        'max_consecutive_losses', 'cooldown_period_bars', 'adx_threshold_for_short', 
        'price_breakdown_period', 'rsi_momentum_threshold'
    ]
    base_values = [pr.get(k, [None]) for k in base_keys]
    
    for combo_parts in itertools.product(itertools.product(*base_values), *filter_options.values()):
        combo = {}
        base_combo_values = combo_parts[0]
        filter_dicts = combo_parts[1:]
        
        combo.update(dict(zip(base_keys, base_combo_values)))
        for d in filter_dicts: combo.update(d)
            
        if combo['ema_short_h1'] >= combo['ema_long_h1']: continue

        if 'FixedRR' in pr['exit_strategy_type']:
            for rr in pr['risk_reward_ratio']:
                final_combo = combo.copy()
                final_combo.update({'exit_strategy_type': 'FixedRR', 'risk_reward_ratio': rr, 'trailing_atr_period': pr['trailing_atr_period'][0], 'trailing_atr_multiplier': pr['trailing_atr_multiplier'][0]})
                yield final_combo
        
        if 'TrailingATR' in pr['exit_strategy_type']:
            for t_per, t_mul in itertools.product(pr['trailing_atr_period'], pr['trailing_atr_multiplier']):
                final_combo = combo.copy()
                final_combo.update({'exit_strategy_type': 'TrailingATR', 'trailing_atr_period': t_per, 'trailing_atr_multiplier': t_mul, 'risk_reward_ratio': pr['risk_reward_ratio'][0]})
                yield final_combo

# ❗ [최종 수정] 커널 내부 로직 강화
@cuda.jit(device=True)
def check_short_entry_conditions_gpu(current_idx, close_prices, low_prices, h1_indicators,
                                     ema_short_idx, ema_long_idx, adx_idx, rsi_idx,
                                     adx_threshold_for_short, price_breakdown_period, rsi_momentum_threshold):
    if current_idx < price_breakdown_period: return False
    if ema_short_idx < 0 or ema_long_idx < 0: return False

    is_trend_bearish = h1_indicators[ema_short_idx, current_idx] < h1_indicators[ema_long_idx, current_idx]

    is_trend_strong = False
    if adx_idx >= 0:
        is_trend_strong = h1_indicators[adx_idx, current_idx] > adx_threshold_for_short
    else:
        is_trend_strong = True

    min_low_in_period = low_prices[current_idx - price_breakdown_period]
    for k in range(1, price_breakdown_period):
        if low_prices[current_idx - price_breakdown_period + k] < min_low_in_period:
            min_low_in_period = low_prices[current_idx - price_breakdown_period + k]
    is_price_breakdown = close_prices[current_idx] < min_low_in_period

    is_momentum_bearish = False
    if rsi_idx >= 0:
        is_momentum_bearish = h1_indicators[rsi_idx, current_idx] < rsi_momentum_threshold
    else:
        is_momentum_bearish = True

    return is_trend_bearish and is_trend_strong and is_price_breakdown and is_momentum_bearish


@cuda.jit
def run_batch_backtest_gpu_kernel(
    open_prices_all, high_prices_all, low_prices_all, close_prices_all, volume_all,
    h1_indicators_all, h4_indicators_all, hour_of_day_all, day_of_week_all,
    param_ema_short_h1_values, param_ema_long_h1_values, param_ema_htf_values,
    param_adx_period_values, param_adx_threshold_values, param_atr_period_sl_values,
    param_atr_multiplier_sl_values, param_risk_reward_ratio_values,
    param_use_htf_ema_filter_flags, param_use_adx_filter_flags, param_risk_per_trade_percentage_values,
    param_exit_strategy_type_codes, param_trailing_atr_period_values, param_trailing_atr_multiplier_values,
    param_use_volume_filter_flags, param_volume_sma_period_values, param_use_rsi_filter_flags,
    param_rsi_period_values, param_rsi_threshold_long_values, param_rsi_threshold_short_values,
    param_use_regime_filter_flags, # ❗ [최종 수정] 커널 시그니처에 존재
    param_adx_threshold_regime_values, param_atr_percent_threshold_regime_values,
    param_time_stop_period_hours_values, param_profit_threshold_for_trail_values,
    param_max_consecutive_losses_values, param_cooldown_period_bars_values,
    param_adx_threshold_for_short_values, param_price_breakdown_period_values,
    param_rsi_momentum_threshold_values,
    h1_ema_short_indices, h1_ema_long_indices, h4_ema_htf_indices, h1_adx_indices,
    h1_atr_sl_indices, h1_atr_trail_indices, h1_vol_sma_indices, h1_rsi_indices,
    out_final_balances, out_total_trades, out_win_trades, out_error_flags,
    out_pnl_percentages, out_profit_factors, out_max_drawdowns,
    data_len, initial_balance_global, commission_global, slippage_global,
    min_trade_size_btc_global, quantity_precision_global,
    allowed_hours_bool_global, blocked_days_bool_global
):
    combo_idx = cuda.grid(1)
    if combo_idx >= len(param_ema_short_h1_values): return

    # 파라미터 로드
    use_regime_filter_f = param_use_regime_filter_flags[combo_idx]
    adx_threshold_regime_p = param_adx_threshold_regime_values[combo_idx]
    atr_percent_threshold_regime_p = param_atr_percent_threshold_regime_values[combo_idx]
    use_htf_ema_filter_f = param_use_htf_ema_filter_flags[combo_idx]
    use_adx_filter_f = param_use_adx_filter_flags[combo_idx]
    use_volume_filter_f = param_use_volume_filter_flags[combo_idx]
    use_rsi_filter_f = param_use_rsi_filter_flags[combo_idx]
    adx_threshold_p = param_adx_threshold_values[combo_idx]
    rsi_threshold_long_p = param_rsi_threshold_long_values[combo_idx]
    rsi_threshold_short_p = param_rsi_threshold_short_values[combo_idx]
    exit_strategy_type_c = param_exit_strategy_type_codes[combo_idx]
    risk_per_trade_percentage_p = param_risk_per_trade_percentage_values[combo_idx]
    atr_multiplier_sl_p = param_atr_multiplier_sl_values[combo_idx]
    risk_reward_ratio_p = param_risk_reward_ratio_values[combo_idx]
    trailing_atr_multiplier_p = param_trailing_atr_multiplier_values[combo_idx]
    time_stop_period_hours_p = param_time_stop_period_hours_values[combo_idx]
    profit_threshold_for_trail_p = param_profit_threshold_for_trail_values[combo_idx]
    max_consecutive_losses_p = param_max_consecutive_losses_values[combo_idx]
    cooldown_period_bars_p = param_cooldown_period_bars_values[combo_idx]
    adx_threshold_for_short_p = param_adx_threshold_for_short_values[combo_idx]
    price_breakdown_period_p = param_price_breakdown_period_values[combo_idx]
    rsi_momentum_threshold_p = param_rsi_momentum_threshold_values[combo_idx]

    # 지표 인덱스 로드
    idx_h1_ema_short, idx_h1_ema_long = h1_ema_short_indices[combo_idx], h1_ema_long_indices[combo_idx]
    idx_h4_ema_htf, idx_h1_adx = h4_ema_htf_indices[combo_idx], h1_adx_indices[combo_idx]
    idx_h1_atr_sl, idx_h1_atr_trail = h1_atr_sl_indices[combo_idx], h1_atr_trail_indices[combo_idx]
    idx_h1_vol_sma, idx_h1_rsi = h1_vol_sma_indices[combo_idx], h1_rsi_indices[combo_idx]

    # 상태 변수
    balance, position = initial_balance_global, POSITION_NONE_GPU
    entry_price_val, position_size_val = 0.0, 0.0
    initial_stop_loss_val, current_stop_loss_val, take_profit_order_val = 0.0, 0.0, 0.0
    entry_idx_val, trade_count_local, win_count_local, error_flag_local = -1, 0, 0, 0
    peak_balance_local, max_drawdown_local = initial_balance_global, 0.0
    gross_profit_local, gross_loss_local = 0.0, 0.0
    consecutive_losses_local, is_in_cooldown_local, cooldown_release_time_idx_local = 0, 0, -1

    # 메인 루프
    for i in range(1, data_len):
        curr_open, curr_high, curr_low, curr_close, curr_volume = open_prices_all[i], high_prices_all[i], low_prices_all[i], close_prices_all[i], volume_all[i]
        
        # 포지션 청산
        if position != POSITION_NONE_GPU:
            exit_reason_code, exit_price = EXIT_REASON_NONE_GPU, 0.0
            current_profit_usd = (curr_close - entry_price_val) * position_size_val if position == POSITION_LONG_GPU else (entry_price_val - curr_close) * position_size_val
            if i - entry_idx_val >= time_stop_period_hours_p and current_profit_usd < 0:
                exit_price, exit_reason_code = curr_close, EXIT_REASON_TIME_STOP_GPU
            if exit_reason_code == EXIT_REASON_NONE_GPU and exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and current_profit_usd >= profit_threshold_for_trail_p:
                if idx_h1_atr_trail >= 0:
                    curr_atr_trail = h1_indicators_all[idx_h1_atr_trail, i]
                    if not math.isnan(curr_atr_trail):
                        if position == POSITION_LONG_GPU: current_stop_loss_val = max(current_stop_loss_val, curr_high - (curr_atr_trail * trailing_atr_multiplier_p))
                        else: current_stop_loss_val = min(current_stop_loss_val, curr_low + (curr_atr_trail * trailing_atr_multiplier_p))
            if exit_reason_code == EXIT_REASON_NONE_GPU:
                if position == POSITION_LONG_GPU:
                    if curr_low <= current_stop_loss_val: exit_price, exit_reason_code = current_stop_loss_val, EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                    elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_high >= take_profit_order_val: exit_price, exit_reason_code = take_profit_order_val, EXIT_REASON_TP_GPU
                else:
                    if curr_high >= current_stop_loss_val: exit_price, exit_reason_code = current_stop_loss_val, EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                    elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_low <= take_profit_order_val: exit_price, exit_reason_code = take_profit_order_val, EXIT_REASON_TP_GPU
            if exit_reason_code != EXIT_REASON_NONE_GPU:
                gross_pnl = (exit_price - entry_price_val) * position_size_val if position == POSITION_LONG_GPU else (entry_price_val - exit_price) * position_size_val
                net_pnl = gross_pnl - (entry_price_val * position_size_val + exit_price * position_size_val) * (commission_global + slippage_global)
                balance += net_pnl
                if balance > peak_balance_local: peak_balance_local = balance
                drawdown = (peak_balance_local - balance) / peak_balance_local if peak_balance_local > 0 else 0.0
                if drawdown > max_drawdown_local: max_drawdown_local = drawdown
                if gross_pnl > 0: gross_profit_local += gross_pnl
                else: gross_loss_local += abs(gross_pnl)
                trade_count_local += 1
                if net_pnl > 0: win_count_local += 1; consecutive_losses_local = 0
                else: consecutive_losses_local += 1
                if consecutive_losses_local >= max_consecutive_losses_p: is_in_cooldown_local, cooldown_release_time_idx_local = 1, i + cooldown_period_bars_p
                if balance <= 0: error_flag_local = 1; break
                position = POSITION_NONE_GPU

        # 포지션 진입
        if position == POSITION_NONE_GPU and error_flag_local == 0:
            if is_in_cooldown_local == 1:
                if i >= cooldown_release_time_idx_local: is_in_cooldown_local, consecutive_losses_local = 0, 0
                else: continue
            if allowed_hours_bool_global[hour_of_day_all[i]] == 0 or blocked_days_bool_global[day_of_week_all[i]] == 1: continue
            
            # ✅ 패치 2: 레짐 게이트 완화 (AND -> OR)
            if use_regime_filter_f:
                if idx_h1_adx >= 0 and idx_h1_atr_sl >= 0:
                    curr_adx_regime, curr_atr_sl = h1_indicators_all[idx_h1_adx, i], h1_indicators_all[idx_h1_atr_sl, i]
                    if not (math.isnan(curr_adx_regime) or math.isnan(curr_atr_sl)):
                        atr_percent = (curr_atr_sl / curr_close) * 100 if curr_close > 0 else 0
                        if not ((curr_adx_regime >= adx_threshold_regime_p) or (atr_percent >= atr_percent_threshold_regime_p)):
                            continue
                # ADX, ATR 인덱스 중 하나라도 없으면 필터 통과 (안전 모드)
            
            if idx_h1_ema_short < 0 or idx_h1_ema_long < 0: continue
            curr_ema_short, prev_ema_short = h1_indicators_all[idx_h1_ema_short, i], h1_indicators_all[idx_h1_ema_short, i-1]
            curr_ema_long, prev_ema_long = h1_indicators_all[idx_h1_ema_long, i], h1_indicators_all[idx_h1_ema_long, i-1]
            if math.isnan(curr_ema_short) or math.isnan(prev_ema_short) or math.isnan(curr_ema_long) or math.isnan(prev_ema_long): continue

            long_signal = prev_ema_short < prev_ema_long and curr_ema_short > curr_ema_long
            short_signal = prev_ema_short > prev_ema_long and curr_ema_short < curr_ema_long and \
                         check_short_entry_conditions_gpu(i, close_prices_all, low_prices_all, h1_indicators_all, idx_h1_ema_short, idx_h1_ema_long, idx_h1_adx, idx_h1_rsi, adx_threshold_for_short_p, price_breakdown_period_p, rsi_momentum_threshold_p)

            if long_signal or short_signal:
                # ✅ 패치 1: H4 EMA NaN은 "중립" 처리
                if use_htf_ema_filter_f and idx_h4_ema_htf >= 0:
                    curr_ema_htf = h4_indicators_all[idx_h4_ema_htf, i]
                    if not math.isnan(curr_ema_htf):
                        if (long_signal and curr_close < curr_ema_htf) or (short_signal and curr_close > curr_ema_htf):
                            long_signal, short_signal = False, False
                
                if use_adx_filter_f and idx_h1_adx >= 0 and h1_indicators_all[idx_h1_adx, i] < adx_threshold_p: long_signal, short_signal = False, False
                if use_volume_filter_f and idx_h1_vol_sma >= 0 and curr_volume <= h1_indicators_all[idx_h1_vol_sma, i]: long_signal, short_signal = False, False
                if use_rsi_filter_f and idx_h1_rsi >= 0:
                    curr_rsi = h1_indicators_all[idx_h1_rsi, i]
                    if (long_signal and curr_rsi < rsi_threshold_long_p) or (short_signal and curr_rsi > rsi_threshold_short_p): long_signal, short_signal = False, False
            
            signal_type = POSITION_LONG_GPU if long_signal else (POSITION_SHORT_GPU if short_signal else POSITION_NONE_GPU)
            if signal_type != POSITION_NONE_GPU:
                if idx_h1_atr_sl < 0: continue
                curr_atr_sl = h1_indicators_all[idx_h1_atr_sl, i]
                if math.isnan(curr_atr_sl) or curr_atr_sl <= 0: continue
                sl_distance = curr_atr_sl * atr_multiplier_sl_p
                pos_size = math.floor((balance * risk_per_trade_percentage_p / sl_distance) * (10**quantity_precision_global)) / (10**quantity_precision_global)
                if pos_size >= min_trade_size_btc_global:
                    position, entry_price_val, position_size_val = signal_type, curr_close, pos_size
                    if signal_type == POSITION_LONG_GPU:
                        initial_stop_loss_val = curr_close - sl_distance
                        if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU: take_profit_order_val = curr_close + (sl_distance * risk_reward_ratio_p)
                    else:
                        initial_stop_loss_val = curr_close + sl_distance
                        if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU: take_profit_order_val = curr_close - (sl_distance * risk_reward_ratio_p)
                    current_stop_loss_val, entry_idx_val = initial_stop_loss_val, i

    out_final_balances[combo_idx], out_total_trades[combo_idx], out_win_trades[combo_idx], out_error_flags[combo_idx] = balance, trade_count_local, win_count_local, error_flag_local
    out_pnl_percentages[combo_idx] = (balance - initial_balance_global) / initial_balance_global * 100 if initial_balance_global > 0 else 0.0
    out_profit_factors[combo_idx] = gross_profit_local / gross_loss_local if gross_loss_local > 0 else (float('inf') if gross_profit_local > 0 else 0.0)
    out_max_drawdowns[combo_idx] = max_drawdown_local * 100

def print_performance_report(performance, params_id="N/A"):
    logger.info(f"\n--- 백테스팅 성과 보고서 (ID: {params_id}) ---")
    if not performance or ("error" in performance and performance["error"]): logger.error(f"성과 데이터 없음 또는 오류 발생 (ID: {params_id})"); return
    report_map = {"initial_balance": "Initial Balance", "final_balance": "Final Balance", "total_net_pnl": "Total Net Pnl", "total_net_pnl_percentage": "Total Net Pnl Percentage", "num_trades": "Num Trades", "num_wins": "Num Wins", "num_losses": "Num Losses", "win_rate_percentage": "Win Rate Percentage", "profit_factor": "Profit Factor", "max_drawdown_percentage": "Max Drawdown Percentage"}
    report_lines = [f"{display_name:<28}: {performance.get(key, 'N/A')}" for key, display_name in report_map.items()]
    logger.info("\n".join(report_lines) + f"\n--- 보고서 종료 (ID: {params_id}) ---\n")

# --- Main Execution Logic ---
if __name__ == "__main__":
    logger.info("=== EMA Crossover 전략 백테스터 V6.4 (최종 안정화 패치) 실행 시작 ===")
    overall_start_time = time.time()
    try:
        if not cuda.is_available() or len(cuda.gpus) == 0: logger.error("CUDA 사용 불가 또는 GPU 없음. 실행 중단."); exit()
        logger.info(f"CUDA 사용 가능. GPU 장치 수: {len(cuda.gpus)}")
        selected_gpu = cuda.get_current_device(); logger.info(f"선택된 GPU: {selected_gpu.name.decode()}")
        cp.cuda.runtime.getDeviceCount(); logger.info("CuPy도 CUDA 장치 인식 완료.")
    except Exception as e: logger.error(f"CUDA/CuPy 초기화 오류: {e}."); exit()

    # 데이터 로딩
    hist_df_primary = get_historical_data(symbol_backtest, interval_primary_bt, start_str=start_date_str, end_str=end_date_str)
    if hist_df_primary is None or hist_df_primary.empty: logger.error("주 시간봉 데이터 로딩 실패."); exit()
    data_len = len(hist_df_primary)
    df_timestamps_for_results = hist_df_primary.index

    # HTF 데이터 로딩
    hist_df_htf = get_historical_data(symbol_backtest, interval_htf_bt, start_str=start_date_str, end_str=end_date_str)
    if hist_df_htf is not None:
        hist_df_htf = hist_df_htf.reindex(hist_df_primary.index, method='ffill')

    # 지표 사전 계산
    unique_indicators, indicator_map, _ = get_unique_indicator_params_and_map(param_ranges)
    master_h1_indicators_np = precompute_all_indicators_for_gpu(hist_df_primary, unique_indicators, "H1", data_len)
    master_h4_indicators_np = precompute_all_indicators_for_gpu(hist_df_htf, unique_indicators, "H4", data_len)
    
    # ❗ [최종 수정] H4 데이터 로딩 실패 시 방어 코드
    if master_h4_indicators_np is None:
        logger.warning("H4 지표를 생성할 수 없어 빈(NaN) 값으로 대체합니다.")
        _, _, num_total_series = get_unique_indicator_params_and_map(param_ranges)
        master_h4_indicators_np = np.full((num_total_series, data_len), np.nan, dtype=np.float64)

    # ✅ 패치 3: 시간 필터 해제 (테스트용)
    logger.info("패치 적용: 모든 시간/요일 거래를 허용합니다.")
    allowed_hours_fixed = set(range(24))
    blocked_days_fixed = set()

    # GPU로 보낼 OHLCV 데이터를 Numpy 배열로 미리 추출
    open_p_np = hist_df_primary['Open'].to_numpy(dtype=np.float64)
    high_p_np = hist_df_primary['High'].to_numpy(dtype=np.float64)
    low_p_np = hist_df_primary['Low'].to_numpy(dtype=np.float64)
    close_p_np = hist_df_primary['Close'].to_numpy(dtype=np.float64)
    volume_np = hist_df_primary['Volume'].to_numpy(dtype=np.float64)

    hour_of_day_np = np.array([ts.hour for ts in df_timestamps_for_results], dtype=np.int8)
    day_of_week_np = np.array([ts.weekday() for ts in df_timestamps_for_results], dtype=np.int8)
    allowed_hours_bool_np = np.array([1 if h in allowed_hours_fixed else 0 for h in range(24)], dtype=np.int8)
    
    # ❗ [최종 수정] 요일 필터 로직 간소화
    blocked_days_bool_np = np.zeros(7, dtype=np.int8)
    
    # 원본 데이터프레임 삭제
    del hist_df_primary, hist_df_htf; gc.collect()

   # 파라미터 조합 생성 (제한 없음)
    logger.info("V6.3: 완전 자동 탐색을 위한 파라미터 조합 생성 시작...")
    prep_start_time = time.time()
    final_param_combos_dicts = list(iter_dependent_param_combos(param_ranges))
    num_final_combinations = len(final_param_combos_dicts)
    
    if num_final_combinations == 0: logger.error("유효한 최종 파라미터 조합 없음."); exit()
    logger.info(f"총 {num_final_combinations}개의 유효 파라미터 조합 생성 완료. ({time.time() - prep_start_time:.2f} 초)")

    # GPU 전송을 위한 리스트 준비
    final_param_ids = []
    final_p_ema_s_vals, final_p_ema_l_vals, final_p_ema_h_vals = [], [], []
    final_p_adx_p_vals, final_p_adx_t_vals = [], []
    final_p_atr_p_sl_vals, final_p_atr_m_sl_vals = [], []
    final_p_rr_vals, final_p_use_htf_f_vals, final_p_use_adx_f_vals = [], [], []
    final_p_risk_vals, final_p_exit_type_vals = [], []
    final_p_trail_atr_p_vals, final_p_trail_atr_m_vals = [], []
    final_p_use_vol_f_vals, final_p_vol_sma_p_vals = [], []
    final_p_use_rsi_f_vals, final_p_rsi_p_vals, final_p_rsi_l_vals, final_p_rsi_s_vals = [], [], [], []
    final_p_use_regime_f_vals = [] # ❗ [최종 수정] 누락된 리스트 추가
    final_p_adx_t_regime_vals, final_p_atr_p_regime_vals = [], []
    final_p_time_stop_h_vals, final_p_profit_trail_vals = [], []
    final_p_max_consec_l_vals, final_p_cooldown_b_vals = [], []
    final_p_adx_t_short_vals, final_p_price_bd_p_vals, final_p_rsi_mom_t_vals = [], [], []
    final_idx_h1_ema_s, final_idx_h1_ema_l, final_idx_h4_ema_h = [], [], []
    final_idx_h1_adx, final_idx_h1_atr_sl, final_idx_h1_atr_trail = [], [], []
    final_idx_h1_vol_sma, final_idx_h1_rsi = [], []

    for combo in final_param_combos_dicts:
        c_id = combo
        param_id_parts = [f"EMA{c_id['ema_short_h1']}-{c_id['ema_long_h1']}"]
        if c_id['use_adx_filter']: param_id_parts.append(f"ADX{c_id['adx_period']}_{c_id['adx_threshold']}")
        if c_id['use_htf_ema_filter']: param_id_parts.append(f"H4Filt{c_id['ema_htf']}")
        param_id_parts.append(f"SL{c_id['atr_multiplier_sl']}ATR{c_id['atr_period_sl']}")
        if c_id['exit_strategy_type'] == 'FixedRR': param_id_parts.append(f"RR{c_id['risk_reward_ratio']}")
        else: param_id_parts.append(f"Trail{c_id['trailing_atr_multiplier']}ATR{c_id['trailing_atr_period']}")
        if c_id['use_volume_filter']: param_id_parts.append(f"VolFilt{c_id['volume_sma_period']}")
        if c_id['use_rsi_filter']: param_id_parts.append(f"RSI{c_id['rsi_period']}_{c_id['rsi_threshold_long']}-{c_id['rsi_threshold_short']}")
        param_id_parts.append(f"Risk{c_id['risk_per_trade_percentage']*100:.1f}%")
        final_param_ids.append("_".join(map(str, param_id_parts)))

        final_p_ema_s_vals.append(combo['ema_short_h1']); final_p_ema_l_vals.append(combo['ema_long_h1']); final_p_ema_h_vals.append(combo['ema_htf'])
        final_p_adx_p_vals.append(combo['adx_period']); final_p_adx_t_vals.append(combo['adx_threshold']); final_p_atr_p_sl_vals.append(combo['atr_period_sl']); final_p_atr_m_sl_vals.append(combo['atr_multiplier_sl'])
        final_p_rr_vals.append(combo['risk_reward_ratio']); final_p_use_htf_f_vals.append(1 if combo.get('use_htf_ema_filter') else 0); final_p_use_adx_f_vals.append(1 if combo.get('use_adx_filter') else 0)
        final_p_risk_vals.append(combo['risk_per_trade_percentage']); final_p_exit_type_vals.append(EXIT_STRATEGY_FIXED_RR_GPU if combo['exit_strategy_type'] == 'FixedRR' else EXIT_STRATEGY_TRAILING_ATR_GPU)
        final_p_trail_atr_p_vals.append(combo['trailing_atr_period']); final_p_trail_atr_m_vals.append(combo['trailing_atr_multiplier'])
        final_p_use_vol_f_vals.append(1 if combo.get('use_volume_filter') else 0); final_p_vol_sma_p_vals.append(combo['volume_sma_period'])
        final_p_use_rsi_f_vals.append(1 if combo.get('use_rsi_filter') else 0); final_p_rsi_p_vals.append(combo['rsi_period']); final_p_rsi_l_vals.append(combo['rsi_threshold_long']); final_p_rsi_s_vals.append(combo['rsi_threshold_short'])
        final_p_use_regime_f_vals.append(1 if combo.get('use_regime_filter') else 0) # ❗ [최종 수정] 누락된 값 추가
        final_p_adx_t_regime_vals.append(combo['adx_threshold_regime']); final_p_atr_p_regime_vals.append(combo['atr_percent_threshold_regime'])
        final_p_time_stop_h_vals.append(combo['time_stop_period_hours']); final_p_profit_trail_vals.append(combo['profit_threshold_for_trail'])
        final_p_max_consec_l_vals.append(combo['max_consecutive_losses']); final_p_cooldown_b_vals.append(combo['cooldown_period_bars'])
        final_p_adx_t_short_vals.append(combo['adx_threshold_for_short']); final_p_price_bd_p_vals.append(combo['price_breakdown_period']); final_p_rsi_mom_t_vals.append(combo['rsi_momentum_threshold'])
        final_idx_h1_ema_s.append(indicator_map.get(('ema', combo['ema_short_h1']), -1))
        final_idx_h1_ema_l.append(indicator_map.get(('ema', combo['ema_long_h1']), -1))
        final_idx_h4_ema_h.append(indicator_map.get(('ema', combo['ema_htf']), -1))
        final_idx_h1_adx.append(indicator_map.get(('adx', combo['adx_period']), -1))
        final_idx_h1_atr_sl.append(indicator_map.get(('atr', combo['atr_period_sl']), -1))
        final_idx_h1_atr_trail.append(indicator_map.get(('atr', combo['trailing_atr_period']), -1))
        final_idx_h1_vol_sma.append(indicator_map.get(('vol_sma', combo['volume_sma_period']), -1))
        final_idx_h1_rsi.append(indicator_map.get(('rsi', combo['rsi_period']), -1))

    prep_duration = time.time() - prep_start_time
    logger.info(f"파라미터 준비 완료. 소요 시간: {prep_duration:.2f} 초")

    # --- GPU 백테스팅 실행 ---
    logger.info("공통 데이터를 GPU로 전송 중...")
    d_open_p=cuda.to_device(open_p_np); d_high_p=cuda.to_device(high_p_np); d_low_p=cuda.to_device(low_p_np); d_close_p=cuda.to_device(close_p_np); d_volume=cuda.to_device(volume_np)
    d_hour_of_day_all=cuda.to_device(hour_of_day_np); d_day_of_week_all=cuda.to_device(day_of_week_np)
    d_allowed_hours_bool_global=cuda.to_device(allowed_hours_bool_np); d_blocked_days_bool_global=cuda.to_device(blocked_days_bool_np)
    d_h1_indicators=cuda.to_device(master_h1_indicators_np); d_h4_indicators = cuda.to_device(master_h4_indicators_np)
    del master_h1_indicators_np, master_h4_indicators_np, open_p_np, high_p_np, low_p_np, close_p_np, volume_np; gc.collect()

    # Memory-safe: maintain only top-K summaries in a min-heap instead of storing all
    TOP_K = int(os.getenv('RESULTS_TOP_K', '1000'))
    topk_heap = []  # items: (metric, seq, summary_dict)
    _seq = 0
    num_batches = math.ceil(num_final_combinations / BATCH_SIZE)
    logger.info(f"총 {num_batches}개의 배치로 나누어 GPU 백테스팅 실행...")

    for batch_num in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, num_final_combinations)
        current_batch_size = end_idx - start_idx
        if current_batch_size <= 0: continue
        logger.info(f"--- 배치 {batch_num + 1}/{num_batches} (조합 {start_idx}-{end_idx-1}, 크기: {current_batch_size}) 처리 시작 ---")
        
        np_p_ema_s_vals_batch=np.array(final_p_ema_s_vals[start_idx:end_idx],dtype=np.int32); np_p_ema_l_vals_batch=np.array(final_p_ema_l_vals[start_idx:end_idx],dtype=np.int32); np_p_ema_h_vals_batch=np.array(final_p_ema_h_vals[start_idx:end_idx],dtype=np.int32)
        np_p_adx_p_vals_batch=np.array(final_p_adx_p_vals[start_idx:end_idx],dtype=np.int32); np_p_adx_t_vals_batch=np.array(final_p_adx_t_vals[start_idx:end_idx],dtype=np.float64); np_p_atr_p_sl_vals_batch=np.array(final_p_atr_p_sl_vals[start_idx:end_idx],dtype=np.int32)
        np_p_atr_m_sl_vals_batch=np.array(final_p_atr_m_sl_vals[start_idx:end_idx],dtype=np.float64); np_p_rr_vals_batch=np.array(final_p_rr_vals[start_idx:end_idx],dtype=np.float64); np_p_use_htf_f_vals_batch=np.array(final_p_use_htf_f_vals[start_idx:end_idx],dtype=np.int8)
        np_p_use_adx_f_vals_batch=np.array(final_p_use_adx_f_vals[start_idx:end_idx],dtype=np.int8); np_p_risk_vals_batch=np.array(final_p_risk_vals[start_idx:end_idx],dtype=np.float64); np_p_exit_type_vals_batch=np.array(final_p_exit_type_vals[start_idx:end_idx],dtype=np.int8)
        np_p_trail_atr_p_vals_batch=np.array(final_p_trail_atr_p_vals[start_idx:end_idx],dtype=np.int32); np_p_trail_atr_m_vals_batch=np.array(final_p_trail_atr_m_vals[start_idx:end_idx],dtype=np.float64); np_p_use_vol_f_vals_batch=np.array(final_p_use_vol_f_vals[start_idx:end_idx],dtype=np.int8)
        np_p_vol_sma_p_vals_batch=np.array(final_p_vol_sma_p_vals[start_idx:end_idx],dtype=np.int32); np_p_use_rsi_f_vals_batch=np.array(final_p_use_rsi_f_vals[start_idx:end_idx],dtype=np.int8); np_p_rsi_p_vals_batch=np.array(final_p_rsi_p_vals[start_idx:end_idx],dtype=np.int32)
        np_p_rsi_l_vals_batch=np.array(final_p_rsi_l_vals[start_idx:end_idx],dtype=np.float64); np_p_rsi_s_vals_batch=np.array(final_p_rsi_s_vals[start_idx:end_idx],dtype=np.float64)
        np_p_use_regime_f_vals_batch = np.array(final_p_use_regime_f_vals[start_idx:end_idx], dtype=np.int8) # ❗ [최종 수정] 누락된 배열 추가
        np_p_adx_t_regime_vals_batch=np.array(final_p_adx_t_regime_vals[start_idx:end_idx],dtype=np.float64); np_p_atr_p_regime_vals_batch=np.array(final_p_atr_p_regime_vals[start_idx:end_idx],dtype=np.float64)
        np_p_time_stop_h_vals_batch=np.array(final_p_time_stop_h_vals[start_idx:end_idx],dtype=np.int32); np_p_profit_trail_vals_batch=np.array(final_p_profit_trail_vals[start_idx:end_idx],dtype=np.float64)
        np_p_max_consec_l_vals_batch=np.array(final_p_max_consec_l_vals[start_idx:end_idx],dtype=np.int32); np_p_cooldown_b_vals_batch=np.array(final_p_cooldown_b_vals[start_idx:end_idx],dtype=np.int32)
        np_p_adx_t_short_vals_batch=np.array(final_p_adx_t_short_vals[start_idx:end_idx],dtype=np.float64); np_p_price_bd_p_vals_batch=np.array(final_p_price_bd_p_vals[start_idx:end_idx],dtype=np.int32); np_p_rsi_mom_t_vals_batch=np.array(final_p_rsi_mom_t_vals[start_idx:end_idx],dtype=np.float64)
        np_idx_h1_ema_s_batch=np.array(final_idx_h1_ema_s[start_idx:end_idx],dtype=np.int32); np_idx_h1_ema_l_batch=np.array(final_idx_h1_ema_l[start_idx:end_idx],dtype=np.int32); np_idx_h4_ema_h_batch=np.array(final_idx_h4_ema_h[start_idx:end_idx],dtype=np.int32)
        np_idx_h1_adx_batch=np.array(final_idx_h1_adx[start_idx:end_idx],dtype=np.int32); np_idx_h1_atr_sl_batch=np.array(final_idx_h1_atr_sl[start_idx:end_idx],dtype=np.int32); np_idx_h1_atr_trail_batch=np.array(final_idx_h1_atr_trail[start_idx:end_idx],dtype=np.int32)
        np_idx_h1_vol_sma_batch=np.array(final_idx_h1_vol_sma[start_idx:end_idx],dtype=np.int32); np_idx_h1_rsi_batch=np.array(final_idx_h1_rsi[start_idx:end_idx],dtype=np.int32)
        
        d_p_ema_s_vals_batch=cuda.to_device(np_p_ema_s_vals_batch); d_p_ema_l_vals_batch=cuda.to_device(np_p_ema_l_vals_batch); d_p_ema_h_vals_batch=cuda.to_device(np_p_ema_h_vals_batch)
        d_p_adx_p_vals_batch=cuda.to_device(np_p_adx_p_vals_batch); d_p_adx_t_vals_batch=cuda.to_device(np_p_adx_t_vals_batch); d_p_atr_p_sl_vals_batch=cuda.to_device(np_p_atr_p_sl_vals_batch)
        d_p_atr_m_sl_vals_batch=cuda.to_device(np_p_atr_m_sl_vals_batch); d_p_rr_vals_batch=cuda.to_device(np_p_rr_vals_batch); d_p_use_htf_f_vals_batch=cuda.to_device(np_p_use_htf_f_vals_batch)
        d_p_use_adx_f_vals_batch=cuda.to_device(np_p_use_adx_f_vals_batch); d_p_risk_vals_batch=cuda.to_device(np_p_risk_vals_batch); d_p_exit_type_vals_batch=cuda.to_device(np_p_exit_type_vals_batch)
        d_p_trail_atr_p_vals_batch=cuda.to_device(np_p_trail_atr_p_vals_batch); d_p_trail_atr_m_vals_batch=cuda.to_device(np_p_trail_atr_m_vals_batch); d_p_use_vol_f_vals_batch=cuda.to_device(np_p_use_vol_f_vals_batch)
        d_p_vol_sma_p_vals_batch=cuda.to_device(np_p_vol_sma_p_vals_batch); d_p_use_rsi_f_vals_batch=cuda.to_device(np_p_use_rsi_f_vals_batch); d_p_rsi_p_vals_batch=cuda.to_device(np_p_rsi_p_vals_batch)
        d_p_rsi_l_vals_batch=cuda.to_device(np_p_rsi_l_vals_batch); d_p_rsi_s_vals_batch=cuda.to_device(np_p_rsi_s_vals_batch)
        d_p_use_regime_f_vals_batch = cuda.to_device(np_p_use_regime_f_vals_batch) # ❗ [최종 수정] 누락된 디바이스 배열 추가
        d_p_adx_t_regime_vals_batch=cuda.to_device(np_p_adx_t_regime_vals_batch); d_p_atr_p_regime_vals_batch=cuda.to_device(np_p_atr_p_regime_vals_batch)
        d_p_time_stop_h_vals_batch=cuda.to_device(np_p_time_stop_h_vals_batch); d_p_profit_trail_vals_batch=cuda.to_device(np_p_profit_trail_vals_batch)
        d_p_max_consec_l_vals_batch=cuda.to_device(np_p_max_consec_l_vals_batch); d_p_cooldown_b_vals_batch=cuda.to_device(np_p_cooldown_b_vals_batch)
        d_p_adx_t_short_vals_batch=cuda.to_device(np_p_adx_t_short_vals_batch); d_p_price_bd_p_vals_batch=cuda.to_device(np_p_price_bd_p_vals_batch); d_p_rsi_mom_t_vals_batch=cuda.to_device(np_p_rsi_mom_t_vals_batch)
        d_idx_h1_ema_s_batch=cuda.to_device(np_idx_h1_ema_s_batch); d_idx_h1_ema_l_batch=cuda.to_device(np_idx_h1_ema_l_batch); d_idx_h4_ema_h_batch=cuda.to_device(np_idx_h4_ema_h_batch)
        d_idx_h1_adx_batch=cuda.to_device(np_idx_h1_adx_batch); d_idx_h1_atr_sl_batch=cuda.to_device(np_idx_h1_atr_sl_batch); d_idx_h1_atr_trail_batch=cuda.to_device(np_idx_h1_atr_trail_batch)
        d_idx_h1_vol_sma_batch=cuda.to_device(np_idx_h1_vol_sma_batch); d_idx_h1_rsi_batch=cuda.to_device(np_idx_h1_rsi_batch)
        
        d_out_final_balances_batch = cuda.device_array(current_batch_size, dtype=np.float64); d_out_total_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32)
        d_out_win_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32); d_out_error_flags_batch = cuda.device_array(current_batch_size, dtype=np.int8)
        d_out_pnl_percentages_batch = cuda.device_array(current_batch_size, dtype=np.float64); d_out_profit_factors_batch = cuda.device_array(current_batch_size, dtype=np.float64)
        d_out_max_drawdowns_batch = cuda.device_array(current_batch_size, dtype=np.float64)

        threads_per_block = 128
        blocks_per_grid = (current_batch_size + threads_per_block - 1) // threads_per_block
        logger.info(f"배치 {batch_num + 1} GPU 커널 실행 시작... Blocks: {blocks_per_grid}, Threads: {threads_per_block}")
        
        batch_kernel_start = time.time()
        run_batch_backtest_gpu_kernel[blocks_per_grid, threads_per_block](
            d_open_p, d_high_p, d_low_p, d_close_p, d_volume, d_h1_indicators, d_h4_indicators, d_hour_of_day_all, d_day_of_week_all,
            d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch, d_p_adx_p_vals_batch, d_p_adx_t_vals_batch,
            d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch, d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch,
            d_p_risk_vals_batch, d_p_exit_type_vals_batch, d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch,
            d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch, d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch,
            d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch,
            d_p_use_regime_f_vals_batch, # ❗ [최종 수정] 정확한 위치에 커널 인자 추가
            d_p_adx_t_regime_vals_batch, d_p_atr_p_regime_vals_batch,
            d_p_time_stop_h_vals_batch, d_p_profit_trail_vals_batch, d_p_max_consec_l_vals_batch, d_p_cooldown_b_vals_batch,
            d_p_adx_t_short_vals_batch, d_p_price_bd_p_vals_batch, d_p_rsi_mom_t_vals_batch,
            d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch, d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch,
            d_idx_h1_atr_trail_batch, d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch,
            d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch,
            d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch,
            data_len, initial_balance, commission_rate_backtest, slippage_rate_per_trade,
            min_trade_size_btc, quantity_precision_bt, d_allowed_hours_bool_global, d_blocked_days_bool_global
        )
        cuda.synchronize()
        batch_kernel_duration = time.time() - batch_kernel_start
        logger.info(f"배치 {batch_num + 1} GPU 커널 실행 완료 ({batch_kernel_duration:.2f} 초)")
        
        h_final_balances_batch = d_out_final_balances_batch.copy_to_host()
        h_total_trades_batch = d_out_total_trades_batch.copy_to_host()
        h_win_trades_batch = d_out_win_trades_batch.copy_to_host()
        h_error_flags_batch = d_out_error_flags_batch.copy_to_host()
        h_pnl_percentages_batch = d_out_pnl_percentages_batch.copy_to_host()
        h_profit_factors_batch = d_out_profit_factors_batch.copy_to_host()
        h_max_drawdowns_batch = d_out_max_drawdowns_batch.copy_to_host()
        for j in range(current_batch_size):
            global_idx = start_idx + j
            num_trades_for_combo = h_total_trades_batch[j]
            performance_summary = {
                "param_id": final_param_ids[global_idx], "initial_balance": initial_balance,
                "final_balance": round(h_final_balances_batch[j], 2),
                "total_net_pnl": round(h_final_balances_batch[j] - initial_balance, 2),
                "total_net_pnl_percentage": round(h_pnl_percentages_batch[j], 2),
                "num_trades": num_trades_for_combo, "num_wins": h_win_trades_batch[j],
                "num_losses": num_trades_for_combo - h_win_trades_batch[j],
                "win_rate_percentage": round((h_win_trades_batch[j] / num_trades_for_combo) * 100 if num_trades_for_combo > 0 else 0, 2),
                "profit_factor": round(h_profit_factors_batch[j], 2) if h_profit_factors_batch[j] != float('inf') else 'inf',
                "max_drawdown_percentage": round(h_max_drawdowns_batch[j], 2),
                "error": bool(h_error_flags_batch[j])
            }
            # Online best selection and top-K maintenance to avoid large memory usage
            # Define metric for ranking (same as later summary): total_net_pnl_percentage
            metric = performance_summary.get("total_net_pnl_percentage", -float('inf'))
            # Maintain only viable results (>=10 trades and no error) in the heap
            if (not performance_summary["error"]) and (num_trades_for_combo >= 10):
                _seq += 1
                item = (metric, _seq, performance_summary)
                if len(topk_heap) < TOP_K:
                    heapq.heappush(topk_heap, item)
                else:
                    # Push-pop to keep only top-K by metric
                    if topk_heap and metric > topk_heap[0][0]:
                        heapq.heapreplace(topk_heap, item)
        
        # ❗ [최종 수정] 배치별 GPU 메모리 정리 (옵션)
        del d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch
        del d_p_adx_p_vals_batch, d_p_adx_t_vals_batch, d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch
        del d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch, d_p_risk_vals_batch, d_p_exit_type_vals_batch
        del d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch, d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch
        del d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch, d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch
        del d_p_use_regime_f_vals_batch, d_p_adx_t_regime_vals_batch, d_p_atr_p_regime_vals_batch
        del d_p_time_stop_h_vals_batch, d_p_profit_trail_vals_batch, d_p_max_consec_l_vals_batch, d_p_cooldown_b_vals_batch
        del d_p_adx_t_short_vals_batch, d_p_price_bd_p_vals_batch, d_p_rsi_mom_t_vals_batch
        del d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch, d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch, d_idx_h1_atr_trail_batch
        del d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch
        del d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch
        del d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch
        gc.collect()

        batch_duration = time.time() - batch_start_time
        logger.info(f"--- 배치 {batch_num + 1}/{num_batches} 처리 완료 ({batch_duration:.2f} 초) ---")

    # --- 최종 결과 처리 및 요약 ---
    logger.info("최종 결과 처리 및 요약 시작...")
    summary_start_time = time.time()
    best_performer = None
    best_metric_val = -float('inf')
    comparison_metric_key = 'total_net_pnl_percentage'
    # Extract sorted top-K (largest first)
    top_results = [t[2] for t in sorted(topk_heap, key=lambda x: x[0], reverse=True)]
    if top_results:
        cand = top_results[0]
        if isinstance(cand.get(comparison_metric_key, None), (int, float)):
            best_performer = cand
            best_metric_val = cand[comparison_metric_key]
    summary_duration = time.time() - summary_start_time
    logger.info(f"결과 처리 및 요약 완료. 소요 시간: {summary_duration:.2f} 초")
    overall_duration = time.time() - overall_start_time
    logger.info(f"총 실행 시간: {timedelta(seconds=overall_duration)}")
    logger.info(f"총 {num_final_combinations}개 조합 테스트 완료.")

    if top_results:
        summary_df = pd.DataFrame(top_results)
        summary_df.rename(columns={
            'param_id': 'Param ID', 'total_net_pnl_percentage': 'PnL %',
            'num_trades': 'Num Trades', 'win_rate_percentage': 'Win Rate %',
            'profit_factor': 'Profit Factor', 'max_drawdown_percentage': 'MDD %'
        }, inplace=True)
        try:
            summary_df_sort_metric = pd.to_numeric(summary_df['PnL %'], errors='coerce').fillna(-float('inf'))
            summary_df.sort_values(by=summary_df_sort_metric.name, ascending=False, inplace=True, na_position='last')
        except Exception as sort_e:
            logger.warning(f"결과 요약 정렬 오류: {sort_e}.")
            summary_df.sort_values(by='PnL %', ascending=False, inplace=True, key=lambda x: pd.to_numeric(x.astype(str).str.replace('inf', '0'), errors='coerce').fillna(-float('inf')))
        logger.info("\n--- 최종 결과 요약 (상위 20개 또는 전체) ---")
        num_to_display = min(20, len(summary_df))
        cols_to_display = ['Param ID', 'PnL %', 'Num Trades', 'Win Rate %', 'Profit Factor', 'MDD %', 'error']
        display_df = summary_df[cols_to_display].head(num_to_display)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            logger.info(f"\n{display_df.to_string(index=False)}\n")
        if best_performer:
            logger.info(f"\n*** 최종 최고 성과 ({comparison_metric_key} 기준, 최소 10회 거래) ***")
            best_param_id_found = best_performer['param_id']
            original_params_for_best = {}
            try:
                best_idx_in_list = final_param_ids.index(best_param_id_found)
                original_params_for_best = final_param_combos_dicts[best_idx_in_list]
            except ValueError:
                logger.warning(f"최고 성과 ID '{best_param_id_found}' 원본 파라미터 찾기 실패.")
            print_performance_report(best_performer, best_param_id_found)
            if original_params_for_best:
                logger.info(f"    Best Parameters (Original Dict): {original_params_for_best}")
                output_data_for_json = {
                    "optimized_for_is_period_start": start_date_str,
                    "optimized_for_is_period_end": end_date_str,
                    "is_performance": {k: v for k, v in best_performer.items() if k != 'param_id'},
                    "generated_at": datetime.now().isoformat(),
                    "parameters": original_params_for_best,
                    "backtest_param_id": best_param_id_found
                }
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"optimized_params_{symbol_backtest}_{timestamp_str}.json"
                output_dir = "wfa_optimized_params_output"
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                output_filepath = os.path.join(output_dir, output_filename)
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(output_data_for_json, f, ensure_ascii=False, indent=4, default=_json_default)
                    logger.info(f"최적 파라미터 및 성과를 JSON 파일로 저장했습니다: '{output_filepath}'")
                except Exception as e:
                    logger.error(f"최적 파라미터 JSON 파일 저장 중 오류 발생: {e}", exc_info=True)
        else:
            logger.info(f"\n최소 거래 횟수(10회) 및 오류 없음 조건을 만족하는 최고 성과 조합을 찾지 못했습니다.")
    else:
        logger.info("\n테스트 결과가 없습니다.")

    # GPU 메모리 정리
    try:
        del d_open_p, d_high_p, d_low_p, d_close_p, d_volume, d_h1_indicators, d_h4_indicators
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        logger.info("공통 GPU 데이터 및 CuPy 메모리 풀 정리 시도 완료.")
    except Exception as e:
        logger.warning(f"공통 GPU 메모리 정리 중 오류: {e}")

    logger.info("백테스터 V6.4 (최종 안정화 패치) 실행 완료.")
