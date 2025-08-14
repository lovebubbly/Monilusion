# -*- coding: utf-8 -*-
# EMA Crossover Strategy Walk-Forward Analyzer (WFA_batchii.py V1.2 - JSON Export Added)
# Based on batchii.py V5.11 and phase2_plan_20250509
# pip install python-binance pandas numpy python-dotenv pandas_ta openpyxl numba cupy-cuda12x

import os
import time
from dotenv import load_dotenv
from binance.client import Client, BinanceAPIException
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import logging
from datetime import datetime, timedelta, timezone # timezone 추가
import itertools
from numba import cuda
import cupy as cp
import gc
import json # JSON 라이브러리 추가

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

# --- API 키 설정 ---
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')
if not api_key or not api_secret:
    logger.error("오류: API 키 필요!")
    exit()

# --- 클라우드 스토리지 경로 설정 ---
CLOUD_SYNC_DIR_WFA_RESULTS = os.getenv('CLOUD_SYNC_DIR_WFA_RESULTS')
if not CLOUD_SYNC_DIR_WFA_RESULTS:
    logger.warning("경고: .env 파일에 CLOUD_SYNC_DIR_WFA_RESULTS 환경변수가 설정되지 않았습니다. JSON 파일이 현재 디렉토리에 저장됩니다.")
    CLOUD_SYNC_DIR_WFA_RESULTS = "." # 기본값으로 현재 디렉토리 사용
else:
    os.makedirs(CLOUD_SYNC_DIR_WFA_RESULTS, exist_ok=True) # 폴더가 없으면 생성

client = Client(api_key, api_secret)
logger.info("바이낸스 클라이언트 초기화 완료 (백테스팅용).")

# --- 백테스팅 공통 설정 ---
symbol_backtest = 'BTCUSDT'
interval_primary_bt = Client.KLINE_INTERVAL_1HOUR # Binance Client 상수 사용
interval_htf_bt = Client.KLINE_INTERVAL_4HOUR     # Binance Client 상수 사용
initial_balance = 10000
commission_rate_backtest = 0.0005
slippage_rate_per_trade = 0.0002
min_trade_size_btc = 0.001
price_precision_bt = 2
quantity_precision_bt = 3

# --- WFA 설정 ---
wfa_overall_start_date_str = "2023-01-01"
wfa_overall_end_date_str = "2025-05-01"
is_period_months = 6
oos_period_months = 3
min_trades_for_best_param = 10

# --- 테스트할 파라미터 범위 정의 ---
param_ranges = {
    'ema_short_h1': [17, 18, 19],
    'ema_long_h1': [20, 21, 22],
    'use_htf_ema_filter': [False, True], 'ema_htf': [50, 100], # real_M1의 ema_htf와 동일 이름 사용 가정
    'use_adx_filter': [False, True], 'adx_period': [14], 'adx_threshold': [20, 25],
    'atr_period_sl': [12, 14, 16], 'atr_multiplier_sl': [2.6, 2.8, 3.0, 3.2],
    'exit_strategy_type': ['FixedRR', 'TrailingATR'], # FixedRR일 때 risk_reward_ratio 사용
    'risk_reward_ratio': [2.2, 2.5, 2.8, 3.0],
    'trailing_atr_period': [12, 14, 16], 'trailing_atr_multiplier': [2.0, 2.5, 3.0], # TrailingATR일 때 사용
    'use_volume_filter': [False, True], 'volume_sma_period': [20, 30],
    'use_rsi_filter': [True], 'rsi_period': [18, 21, 24],
    'rsi_threshold_long': [48, 50, 52], 'rsi_threshold_short': [43, 45, 47],
    'risk_per_trade_percentage': [0.018, 0.02, 0.022],
    # 'leverage_config'는 real_M1.py의 TRADING_PARAMS에 있지만, oos.py에서는 직접 사용하지 않으므로 JSON 생성 시 기본값 또는 고정값 사용 가능
}

# --- GPU 커널 및 백테스팅 관련 상수 ---
BATCH_SIZE = 500000
POSITION_NONE_GPU = 0; POSITION_LONG_GPU = 1; POSITION_SHORT_GPU = -1
EXIT_REASON_NONE_GPU = -1; EXIT_REASON_SL_GPU = 0; EXIT_REASON_TP_GPU = 1; EXIT_REASON_TRAIL_SL_GPU = 2
EXIT_STRATEGY_FIXED_RR_GPU = 0; EXIT_STRATEGY_TRAILING_ATR_GPU = 1

def get_historical_data(symbol, interval, start_str=None, end_str=None, limit=1000):
    df = pd.DataFrame()
    logger.info(f"데이터 로딩 시작: Symbol={symbol}, Interval={interval}, Start={start_str}, End={end_str}")
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
    if start_time_ms_calc is None or end_time_ms_calc is None :
        logger.error("WFA에는 명확한 시작 및 종료 시간이 필요합니다."); return None

    current_start_time_ms = start_time_ms_calc
    klines_list = []
    while current_start_time_ms < end_time_ms_calc:
        try:
            klines_batch = client.futures_klines(symbol=symbol, interval=interval, startTime=current_start_time_ms, limit=limit, endTime=end_time_ms_calc)
            if not klines_batch: break
            klines_list.extend(klines_batch)
            last_kline_close_time = klines_batch[-1][6]
            current_start_time_ms = last_kline_close_time + 1
        except BinanceAPIException as e: logger.error(f"API 오류 (데이터 로딩): {e}"); time.sleep(5);
        except Exception as e_gen: logger.error(f"알 수 없는 오류 (데이터 로딩): {e_gen}"); break
        time.sleep(0.1)
    if not klines_list: logger.warning(f"{start_str}~{end_str} 기간 데이터 가져오기 실패 또는 데이터 없음."); return None
    df = pd.DataFrame(klines_list, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms', utc=True) # UTC로 명시
    df.set_index('Open time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
    df.drop_duplicates(inplace=True); df.sort_index(inplace=True)
    
    # WFA에서는 정확한 기간 필터링이 중요 (UTC 기준으로 파싱된 datetime 객체 사용)
    start_dt_utc = pd.to_datetime(start_str).tz_localize('UTC') if pd.to_datetime(start_str).tzinfo is None else pd.to_datetime(start_str).tz_convert('UTC')
    # end_str은 해당 날짜의 마지막 시간까지 포함하도록 (23:59:59.999...)
    end_dt_utc = (pd.to_datetime(end_str) + timedelta(days=1) - timedelta(milliseconds=1)).tz_localize('UTC') if pd.to_datetime(end_str).tzinfo is None else (pd.to_datetime(end_str) + timedelta(days=1) - timedelta(milliseconds=1)).tz_convert('UTC')

    df = df[(df.index >= start_dt_utc) & (df.index <= end_dt_utc)]
    if df.empty: logger.warning(f"{start_str}~{end_str} 기간에 해당하는 데이터가 없습니다 (UTC 필터링 후)."); return None
    logger.info(f"총 {len(df)}개 Klines 로드 완료. 기간 (UTC): {df.index.min()} ~ {df.index.max()}")
    return df

def get_unique_indicator_params_and_map(current_param_ranges):
    unique_params_values = {
        'ema_lengths': sorted(list(set(current_param_ranges.get('ema_short_h1', []) +
                                      current_param_ranges.get('ema_long_h1', []) +
                                      current_param_ranges.get('ema_htf', [])))),
        'adx_periods': sorted(list(set(current_param_ranges.get('adx_period', [])))),
        'atr_periods': sorted(list(set(current_param_ranges.get('atr_period_sl', []) +
                                      current_param_ranges.get('trailing_atr_period', [])))),
        'volume_sma_periods': sorted(list(set(current_param_ranges.get('volume_sma_period', [])))),
        'rsi_periods': sorted(list(set(current_param_ranges.get('rsi_period', [])))),
    }
    for key in unique_params_values: # 기본값 설정
        if not unique_params_values[key]:
            if key == 'ema_lengths': unique_params_values[key] = [20]
            elif key == 'adx_periods': unique_params_values[key] = [14]
            elif key == 'atr_periods': unique_params_values[key] = [14]
            elif key == 'volume_sma_periods': unique_params_values[key] = [20]
            elif key == 'rsi_periods': unique_params_values[key] = [14]

    indicator_map = {}; current_idx = 0
    for length in unique_params_values['ema_lengths']: indicator_map[('ema', length)] = current_idx; current_idx += 1
    for period in unique_params_values['atr_periods']: indicator_map[('atr', period)] = current_idx; current_idx += 1
    for period in unique_params_values['adx_periods']:
        indicator_map[('adx', period)] = current_idx; indicator_map[('dmp', period)] = current_idx + 1; indicator_map[('dmn', period)] = current_idx + 2
        current_idx += 3
    for period in unique_params_values['volume_sma_periods']: indicator_map[('vol_sma', period)] = current_idx; current_idx += 1
    for period in unique_params_values['rsi_periods']: indicator_map[('rsi', period)] = current_idx; current_idx += 1
    num_total_indicator_series = current_idx
    logger.debug(f"고유 지표 파라미터 추출 및 매핑 완료. 총 {num_total_indicator_series}개 시리즈.")
    return unique_params_values, indicator_map, num_total_indicator_series

def precompute_all_indicators_for_gpu(df_ohlcv, unique_params_dict, indicator_to_idx_map_local, num_total_series_local, interval_suffix, data_length):
    if df_ohlcv is None or df_ohlcv.empty: logger.warning(f"사전 계산용 {interval_suffix} 데이터 없음."); return None
    master_indicator_array = np.full((num_total_series_local, data_length), np.nan, dtype=np.float64)
    logger.debug(f"{interval_suffix} 데이터용 마스터 지표 배열 생성 ({master_indicator_array.shape}).")
    # ... (기존 지표 계산 로직과 동일) ...
    for length in unique_params_dict.get('ema_lengths', []):
        if length > 0:
            idx = indicator_to_idx_map_local.get(('ema', length))
            if idx is not None:
                try:
                    ema_series = df_ohlcv.ta.ema(length=length, append=False)
                    if ema_series is not None and len(ema_series) == data_length: master_indicator_array[idx, :] = ema_series.to_numpy()
                except Exception as e: logger.error(f"EMA_{length}_{interval_suffix} 계산 오류: {e}")
    for period in unique_params_dict.get('atr_periods', []):
        if period > 0:
            idx = indicator_to_idx_map_local.get(('atr', period))
            if idx is not None:
                try:
                    atr_series = df_ohlcv.ta.atr(length=period, append=False) # high, low, close 필요
                    if atr_series is not None and len(atr_series) == data_length: master_indicator_array[idx, :] = atr_series.to_numpy()
                except Exception as e: logger.error(f"ATR_{period}_{interval_suffix} 계산 오류: {e}")
    for period in unique_params_dict.get('adx_periods', []):
        if period > 0:
            adx_idx = indicator_to_idx_map_local.get(('adx', period)); dmp_idx = indicator_to_idx_map_local.get(('dmp', period)); dmn_idx = indicator_to_idx_map_local.get(('dmn', period))
            if adx_idx is not None and dmp_idx is not None and dmn_idx is not None:
                try:
                    adx_df = df_ohlcv.ta.adx(length=period, append=False) # high, low, close 필요
                    if adx_df is not None:
                        adx_col, dmp_col, dmn_col = f"ADX_{period}", f"DMP_{period}", f"DMN_{period}"
                        if adx_col in adx_df and len(adx_df[adx_col]) == data_length: master_indicator_array[adx_idx, :] = adx_df[adx_col].to_numpy()
                        if dmp_col in adx_df and len(adx_df[dmp_col]) == data_length: master_indicator_array[dmp_idx, :] = adx_df[dmp_col].to_numpy()
                        if dmn_col in adx_df and len(adx_df[dmn_col]) == data_length: master_indicator_array[dmn_idx, :] = adx_df[dmn_col].to_numpy()
                except Exception as e: logger.error(f"ADX/DMP/DMN_{period}_{interval_suffix} 계산 오류: {e}")
    if 'Volume' in df_ohlcv.columns:
        volume_series = df_ohlcv['Volume']
        for period in unique_params_dict.get('volume_sma_periods', []):
            if period > 0:
                idx = indicator_to_idx_map_local.get(('vol_sma', period))
                if idx is not None:
                    try:
                        vol_sma_series = ta.sma(volume_series, length=period, append=False);
                        if vol_sma_series is not None and len(vol_sma_series) == data_length: master_indicator_array[idx, :] = vol_sma_series.to_numpy()
                    except Exception as e: logger.error(f"Volume SMA_{period}_{interval_suffix} 계산 오류: {e}")
    else: logger.warning(f"{interval_suffix} 데이터에 'Volume' 컬럼 없어 Volume SMA 계산 스킵.")
    for period in unique_params_dict.get('rsi_periods', []):
        if period > 0:
            idx = indicator_to_idx_map_local.get(('rsi', period))
            if idx is not None:
                try:
                    rsi_series = df_ohlcv.ta.rsi(length=period, append=False); # close 필요
                    if rsi_series is not None and len(rsi_series) == data_length: master_indicator_array[idx, :] = rsi_series.to_numpy()
                except Exception as e: logger.error(f"RSI_{period}_{interval_suffix} 계산 오류: {e}")
    logger.debug(f"{interval_suffix} 데이터 마스터 지표 배열 계산 완료.")
    return master_indicator_array

@cuda.jit
def run_batch_backtest_gpu_kernel(
    open_prices_all, high_prices_all, low_prices_all, close_prices_all, volume_all,
    h1_indicators_all, h4_indicators_all,
    param_ema_short_h1_values, param_ema_long_h1_values, param_ema_htf_values,
    param_adx_period_values, param_adx_threshold_values,
    param_atr_period_sl_values, param_atr_multiplier_sl_values,
    param_risk_reward_ratio_values,
    param_use_htf_ema_filter_flags, param_use_adx_filter_flags,
    param_risk_per_trade_percentage_values,
    param_exit_strategy_type_codes,
    param_trailing_atr_period_values, param_trailing_atr_multiplier_values,
    param_use_volume_filter_flags, param_volume_sma_period_values,
    param_use_rsi_filter_flags, param_rsi_period_values,
    param_rsi_threshold_long_values, param_rsi_threshold_short_values,
    h1_ema_short_indices, h1_ema_long_indices, h4_ema_htf_indices,
    h1_adx_indices, h1_atr_sl_indices, h1_atr_trail_indices,
    h1_vol_sma_indices, h1_rsi_indices,
    out_final_balances, out_total_trades, out_win_trades, out_error_flags,
    out_pnl_percentages, out_profit_factors, out_max_drawdowns,
    data_len: int, initial_balance_global: float, commission_global: float, slippage_global: float,
    min_trade_size_btc_global: float, quantity_precision_global: int
):
    # ... (기존 GPU 커널 로직과 동일) ...
    combo_idx = cuda.blockIdx.x
    if combo_idx >= len(param_ema_short_h1_values): return
    adx_threshold_p = param_adx_threshold_values[combo_idx]
    atr_multiplier_sl_p = param_atr_multiplier_sl_values[combo_idx]
    risk_reward_ratio_p = param_risk_reward_ratio_values[combo_idx]
    use_htf_ema_filter_f = param_use_htf_ema_filter_flags[combo_idx]
    use_adx_filter_f = param_use_adx_filter_flags[combo_idx]
    risk_per_trade_percentage_p = param_risk_per_trade_percentage_values[combo_idx]
    exit_strategy_type_c = param_exit_strategy_type_codes[combo_idx]
    trailing_atr_multiplier_p = param_trailing_atr_multiplier_values[combo_idx]
    use_volume_filter_f = param_use_volume_filter_flags[combo_idx]
    use_rsi_filter_f = param_use_rsi_filter_flags[combo_idx]
    rsi_threshold_long_p = param_rsi_threshold_long_values[combo_idx]
    rsi_threshold_short_p = param_rsi_threshold_short_values[combo_idx]
    idx_h1_ema_short = h1_ema_short_indices[combo_idx]; idx_h1_ema_long = h1_ema_long_indices[combo_idx]
    idx_h4_ema_htf = h4_ema_htf_indices[combo_idx]; idx_h1_adx = h1_adx_indices[combo_idx]
    idx_h1_atr_sl = h1_atr_sl_indices[combo_idx]; idx_h1_atr_trail = h1_atr_trail_indices[combo_idx]
    idx_h1_vol_sma = h1_vol_sma_indices[combo_idx]; idx_h1_rsi = h1_rsi_indices[combo_idx]
    balance = initial_balance_global; position = POSITION_NONE_GPU; entry_price_val = 0.0; position_size_val = 0.0
    initial_stop_loss_val = 0.0; current_stop_loss_val = 0.0; take_profit_order_val = 0.0; entry_idx_val = -1
    trade_count_local = 0; win_count_local = 0; error_flag_local = 0; peak_balance_local = initial_balance_global
    max_drawdown_local = 0.0; gross_profit_local = 0.0; gross_loss_local = 0.0
    for i in range(1, data_len):
        curr_open = open_prices_all[i]; curr_high = high_prices_all[i]; curr_low = low_prices_all[i]; curr_close = close_prices_all[i]; curr_volume = volume_all[i]
        curr_ema_short_h1 = h1_indicators_all[idx_h1_ema_short, i]; prev_ema_short_h1 = h1_indicators_all[idx_h1_ema_short, i-1]
        curr_ema_long_h1 = h1_indicators_all[idx_h1_ema_long, i]; prev_ema_long_h1 = h1_indicators_all[idx_h1_ema_long, i-1]
        curr_atr_sl = h1_indicators_all[idx_h1_atr_sl, i]; curr_adx = math.nan; curr_ema_htf = math.nan; curr_atr_trail = math.nan; curr_vol_sma = math.nan; curr_rsi = math.nan
        if use_adx_filter_f and idx_h1_adx >= 0: curr_adx = h1_indicators_all[idx_h1_adx, i]
        if use_htf_ema_filter_f and idx_h4_ema_htf >= 0: curr_ema_htf = h4_indicators_all[idx_h4_ema_htf, i] # h4_indicators_all 사용
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and idx_h1_atr_trail >=0: curr_atr_trail = h1_indicators_all[idx_h1_atr_trail, i]
        if use_volume_filter_f and idx_h1_vol_sma >=0: curr_vol_sma = h1_indicators_all[idx_h1_vol_sma, i]
        if use_rsi_filter_f and idx_h1_rsi >=0: curr_rsi = h1_indicators_all[idx_h1_rsi, i]
        if math.isnan(curr_ema_short_h1) or math.isnan(curr_ema_long_h1) or math.isnan(prev_ema_short_h1) or math.isnan(prev_ema_long_h1) or math.isnan(curr_atr_sl) : continue
        if use_adx_filter_f and math.isnan(curr_adx): continue
        if use_htf_ema_filter_f and math.isnan(curr_ema_htf): continue
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and math.isnan(curr_atr_trail): continue
        if use_volume_filter_f and math.isnan(curr_vol_sma): continue
        if use_rsi_filter_f and math.isnan(curr_rsi): continue
        if curr_atr_sl <= 0: continue
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and curr_atr_trail <= 0: continue
        if position != POSITION_NONE_GPU:
            exit_reason_code = EXIT_REASON_NONE_GPU; exit_price = 0.0
            if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU:
                if position == POSITION_LONG_GPU: new_trail_sl = curr_high - (curr_atr_trail * trailing_atr_multiplier_p); current_stop_loss_val = max(current_stop_loss_val, new_trail_sl)
                elif position == POSITION_SHORT_GPU: new_trail_sl = curr_low + (curr_atr_trail * trailing_atr_multiplier_p); current_stop_loss_val = min(current_stop_loss_val, new_trail_sl)
            if position == POSITION_LONG_GPU:
                if curr_low <= current_stop_loss_val: exit_price = current_stop_loss_val; exit_reason_code = EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_high >= take_profit_order_val: exit_price = take_profit_order_val; exit_reason_code = EXIT_REASON_TP_GPU
            elif position == POSITION_SHORT_GPU:
                if curr_high >= current_stop_loss_val: exit_price = current_stop_loss_val; exit_reason_code = EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_low <= take_profit_order_val: exit_price = take_profit_order_val; exit_reason_code = EXIT_REASON_TP_GPU
            if exit_reason_code != EXIT_REASON_NONE_GPU:
                gross_pnl = (exit_price - entry_price_val) * position_size_val if position == POSITION_LONG_GPU else (entry_price_val - exit_price) * position_size_val
                entry_value = entry_price_val * position_size_val; exit_value = exit_price * position_size_val
                entry_cost = entry_value * (commission_global + slippage_global); exit_cost = exit_value * (commission_global + slippage_global)
                total_trade_cost = entry_cost + exit_cost; net_pnl = gross_pnl - total_trade_cost; balance += net_pnl
                if balance > peak_balance_local: peak_balance_local = balance
                drawdown = (peak_balance_local - balance) / peak_balance_local if peak_balance_local > 0 else 0.0
                if drawdown > max_drawdown_local: max_drawdown_local = drawdown
                if gross_pnl > 0: gross_profit_local += gross_pnl
                else: gross_loss_local += abs(gross_pnl)
                trade_count_local += 1
                if net_pnl > 0: win_count_local += 1
                if balance <= 0: error_flag_local = 1; break
                position = POSITION_NONE_GPU; entry_price_val = 0.0; position_size_val = 0.0; initial_stop_loss_val = 0.0
                current_stop_loss_val = 0.0; take_profit_order_val = 0.0; entry_idx_val = -1
        if position == POSITION_NONE_GPU and error_flag_local == 0:
            long_signal, short_signal = False, False
            if prev_ema_short_h1 < prev_ema_long_h1 and curr_ema_short_h1 > curr_ema_long_h1: long_signal = True
            elif prev_ema_short_h1 > prev_ema_long_h1 and curr_ema_short_h1 < curr_ema_long_h1: short_signal = True
            if long_signal or short_signal:
                if use_htf_ema_filter_f:
                    if long_signal and curr_close < curr_ema_htf: long_signal = False
                    if short_signal and curr_close > curr_ema_htf: short_signal = False
                if use_adx_filter_f and curr_adx < adx_threshold_p: long_signal, short_signal = False, False
                if use_volume_filter_f and curr_volume <= curr_vol_sma : long_signal, short_signal = False, False
                if use_rsi_filter_f:
                    if long_signal and curr_rsi < rsi_threshold_long_p: long_signal = False
                    if short_signal and curr_rsi > rsi_threshold_short_p: short_signal = False
            signal_type = POSITION_NONE_GPU
            if long_signal: signal_type = POSITION_LONG_GPU
            elif short_signal: signal_type = POSITION_SHORT_GPU
            if signal_type != POSITION_NONE_GPU:
                entry_p = curr_close; sl_distance = curr_atr_sl * atr_multiplier_sl_p
                if sl_distance <= 0: continue
                potential_initial_sl, potential_tp = 0.0, 0.0
                if signal_type == POSITION_LONG_GPU:
                    potential_initial_sl = entry_p - sl_distance
                    if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU: potential_tp = entry_p + (sl_distance * risk_reward_ratio_p)
                else:
                    potential_initial_sl = entry_p + sl_distance
                    if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU: potential_tp = entry_p - (sl_distance * risk_reward_ratio_p)
                valid_entry_conditions = False
                if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU:
                    if signal_type == POSITION_LONG_GPU and potential_tp > entry_p and potential_initial_sl < entry_p: valid_entry_conditions = True
                    elif signal_type == POSITION_SHORT_GPU and potential_tp < entry_p and potential_initial_sl > entry_p: valid_entry_conditions = True
                elif exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU:
                     if signal_type == POSITION_LONG_GPU and potential_initial_sl < entry_p : valid_entry_conditions = True
                     elif signal_type == POSITION_SHORT_GPU and potential_initial_sl > entry_p : valid_entry_conditions = True
                if valid_entry_conditions:
                    risk_amount_per_trade = balance * risk_per_trade_percentage_p; position_size_calc = risk_amount_per_trade / sl_distance
                    power_factor = 1;
                    for _ in range(quantity_precision_global): power_factor *= 10
                    calculated_size = math.floor(position_size_calc * power_factor) / power_factor
                    if calculated_size >= min_trade_size_btc_global:
                        position = signal_type; entry_price_val = entry_p; position_size_val = calculated_size
                        initial_stop_loss_val = potential_initial_sl; current_stop_loss_val = potential_initial_sl
                        if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU: take_profit_order_val = potential_tp
                        else: take_profit_order_val = 0.0
                        entry_idx_val = i
    out_final_balances[combo_idx] = balance; out_total_trades[combo_idx] = trade_count_local; out_win_trades[combo_idx] = win_count_local
    out_error_flags[combo_idx] = error_flag_local; total_pnl = balance - initial_balance_global
    out_pnl_percentages[combo_idx] = (total_pnl / initial_balance_global) * 100 if initial_balance_global > 0 else 0.0
    out_profit_factors[combo_idx] = gross_profit_local / gross_loss_local if gross_loss_local > 0 else (float('inf') if gross_profit_local > 0 else 0.0)
    out_max_drawdowns[combo_idx] = max_drawdown_local * 100

def print_performance_report(performance, params_id="N/A", context="IS"):
    logger.info(f"\n--- {context} 백테스팅 성과 보고서 (ID: {params_id}) ---")
    if not performance: logger.info("성과 데이터 없음."); return
    if "error" in performance and performance["error"]: logger.error(f"{context} 백테스팅 오류 발생! (ID: {params_id})"); return
    report_map = {
        "initial_balance": "Initial Balance", "final_balance": "Final Balance",
        "total_net_pnl": "Total Net Pnl", "total_net_pnl_percentage": "Total Net Pnl Percentage",
        "num_trades": "Num Trades", "num_wins": "Num Wins", "num_losses": "Num Losses",
        "win_rate_percentage": "Win Rate Percentage", "profit_factor": "Profit Factor",
        "max_drawdown_percentage": "Max Drawdown Percentage"
    }
    report_lines = [f"{display_name:<28}: {performance.get(key, 'N/A')}" for key, display_name in report_map.items()]
    logger.info("\n".join(report_lines))
    logger.info(f"--- {context} 보고서 종료 (ID: {params_id}) ---\n")

def map_interval_to_real_m1_format(binance_client_interval):
    """Binance Client KLINE_INTERVAL 상수를 real_M1.py의 문자열 형식으로 변환합니다."""
    # real_M1.py에서 사용하는 interval 형식에 맞춰 추가/수정 필요
    if binance_client_interval == Client.KLINE_INTERVAL_1MINUTE: return "1m"
    if binance_client_interval == Client.KLINE_INTERVAL_3MINUTE: return "3m"
    if binance_client_interval == Client.KLINE_INTERVAL_5MINUTE: return "5m"
    if binance_client_interval == Client.KLINE_INTERVAL_15MINUTE: return "15m"
    if binance_client_interval == Client.KLINE_INTERVAL_30MINUTE: return "30m"
    if binance_client_interval == Client.KLINE_INTERVAL_1HOUR: return "1h"
    if binance_client_interval == Client.KLINE_INTERVAL_2HOUR: return "2h"
    if binance_client_interval == Client.KLINE_INTERVAL_4HOUR: return "4h"
    if binance_client_interval == Client.KLINE_INTERVAL_6HOUR: return "6h"
    if binance_client_interval == Client.KLINE_INTERVAL_8HOUR: return "8h"
    if binance_client_interval == Client.KLINE_INTERVAL_12HOUR: return "12h"
    if binance_client_interval == Client.KLINE_INTERVAL_1DAY: return "1d"
    if binance_client_interval == Client.KLINE_INTERVAL_3DAY: return "3d"
    if binance_client_interval == Client.KLINE_INTERVAL_1WEEK: return "1w"
    if binance_client_interval == Client.KLINE_INTERVAL_1MONTH: return "1M"
    logger.warning(f"알 수 없는 KLINE_INTERVAL 형식: {binance_client_interval}. 원본 값 반환.")
    return binance_client_interval # 매핑 실패 시 원본 반환

def save_optimized_params_to_json(best_params_dict, performance_summary, wfa_info, output_dir):
    """최적화된 파라미터와 성과 요약을 JSON 파일로 저장합니다."""
    try:
        now_utc = datetime.now(timezone.utc)
        # 파일명 규칙: optimized_params_YYYYMMDD_HHMMSS_IS_START_to_IS_END.json
        filename = f"optimized_params_{now_utc.strftime('%Y%m%d_%H%M%S')}_{wfa_info['is_period_start']}_to_{wfa_info['is_period_end']}.json"
        filepath = os.path.join(output_dir, filename)

        # real_M1.py의 TRADING_PARAMS 형식으로 매핑
        # 이 부분은 실제 real_M1.py의 TRADING_PARAMS 구조와 oos.py의 파라미터 이름을 비교하여 정확하게 맞춰야 함
        trading_params_for_real_m1 = {
            "symbol": symbol_backtest, # oos.py의 전역 변수 사용
            "interval_primary": map_interval_to_real_m1_format(interval_primary_bt), # 형식 변환
            "data_fetch_limit_primary": 200, # real_M1.py의 기본값 또는 WFA에서 최적화하지 않았다면 고정값
            "ema_short_period": best_params_dict.get('ema_short_h1'),
            "ema_long_period": best_params_dict.get('ema_long_h1'),
            "rsi_period": best_params_dict.get('rsi_period'),
            "rsi_threshold_long": best_params_dict.get('rsi_threshold_long'),
            "rsi_threshold_short": best_params_dict.get('rsi_threshold_short'),
            "atr_period_sl": best_params_dict.get('atr_period_sl'),
            "atr_multiplier_sl": best_params_dict.get('atr_multiplier_sl'),
            "risk_reward_ratio": best_params_dict.get('risk_reward_ratio') if best_params_dict.get('exit_strategy_type') == 'FixedRR' else param_ranges['risk_reward_ratio'][0], # FixedRR일 때만 의미 있음
            "use_adx_filter": best_params_dict.get('use_adx_filter', False),
            "adx_period": best_params_dict.get('adx_period'),
            "adx_threshold": best_params_dict.get('adx_threshold'),
            "use_htf_ema_filter": best_params_dict.get('use_htf_ema_filter', False),
            "ema_htf": best_params_dict.get('ema_htf'), # oos.py의 ema_htf와 동일 이름 사용 가정
            "use_volume_filter": best_params_dict.get('use_volume_filter', False),
            "volume_sma_period": best_params_dict.get('volume_sma_period'),
            "risk_per_trade_percentage": best_params_dict.get('risk_per_trade_percentage'),
            "leverage_config": 2, # real_M1.py의 기본값 또는 고정값 (WFA에서 최적화하지 않음)
            # real_M1.py의 TRADING_PARAMS에 있지만 oos.py에서 직접 사용되지 않는 값들은 기본값 또는 고정값으로 채워야 함
            "tick_size": 0.01, # 예시: 실제 값은 initialize_exchange_info()에서 가져오므로 여기서는 기본값
            "min_contract_size": 0.001, # 예시
            "quantity_precision": 3, # 예시
            "price_precision": 2, # 예시
            "min_trade_value_usdt": 5, # 예시
            "db_path": f"trading_data_{symbol_backtest}.sqlite", # 예시
        }
        # 누락된 키가 없도록 주의 (real_M1.py의 TRADING_PARAMS의 모든 키가 포함되어야 함)

        data_to_save = {
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "wfa_window_info": wfa_info,
            "is_performance_summary": performance_summary, # GPU 커널 결과에서 가져온 상세 정보
            "trading_params": trading_params_for_real_m1
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2) # indent로 가독성 높임
        logger.info(f"최적 파라미터 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"JSON 파일 저장 중 오류 발생: {e}", exc_info=True)
        return None

def run_optimization_for_is_period(
    is_df_primary, is_df_htf,
    current_param_ranges, common_indicator_map, common_unique_configs, common_num_total_series,
    is_period_label, wfa_current_info # JSON 저장을 위한 wfa_info 추가
):
    logger.info(f"--- {is_period_label}: 인-샘플 최적화 시작 ---")
    is_data_length = len(is_df_primary)
    if is_data_length == 0:
        logger.warning(f"{is_period_label}: 인-샘플 데이터 없음. 최적화 스킵.")
        return None, None

    is_master_h1_indicators_np = precompute_all_indicators_for_gpu(is_df_primary, common_unique_configs, common_indicator_map, common_num_total_series, f"{is_period_label}_H1", is_data_length)
    is_master_h4_indicators_np = None
    if is_df_htf is not None and not is_df_htf.empty:
        is_master_h4_indicators_np = precompute_all_indicators_for_gpu(is_df_htf, common_unique_configs, common_indicator_map, common_num_total_series, f"{is_period_label}_H4", is_data_length)
    else:
        is_master_h4_indicators_np = np.full_like(is_master_h1_indicators_np, np.nan) if is_master_h1_indicators_np is not None else None

    if is_master_h1_indicators_np is None: logger.error(f"{is_period_label}: H1 마스터 지표 생성 실패."); return None, None
    
    try:
        d_is_open_p=cuda.to_device(is_df_primary['Open'].to_numpy(dtype=np.float64))
        d_is_high_p=cuda.to_device(is_df_primary['High'].to_numpy(dtype=np.float64))
        d_is_low_p=cuda.to_device(is_df_primary['Low'].to_numpy(dtype=np.float64))
        d_is_close_p=cuda.to_device(is_df_primary['Close'].to_numpy(dtype=np.float64))
        d_is_volume=cuda.to_device(is_df_primary['Volume'].to_numpy(dtype=np.float64))
        d_is_h1_indicators=cuda.to_device(is_master_h1_indicators_np)
        d_is_h4_indicators=cuda.to_device(is_master_h4_indicators_np if is_master_h4_indicators_np is not None else d_is_h1_indicators) # H4 없으면 H1으로 채움 (커널에서 사용 안함)

        param_keys, param_values_iter_list = zip(*current_param_ranges.items())
        param_generator = itertools.product(*param_values_iter_list)
        final_param_combos_dicts = []; final_param_ids = []
        # ... (기존 파라미터 리스트 초기화와 동일) ...
        final_p_ema_s_vals, final_p_ema_l_vals, final_p_ema_h_vals = [], [], []
        final_p_adx_p_vals, final_p_adx_t_vals = [], []; final_p_atr_p_sl_vals, final_p_atr_m_sl_vals = [], []
        final_p_rr_vals, final_p_use_htf_f_vals, final_p_use_adx_f_vals = [], [], []
        final_p_risk_vals, final_p_exit_type_vals = [], []; final_p_trail_atr_p_vals, final_p_trail_atr_m_vals = [], []
        final_p_use_vol_f_vals, final_p_vol_sma_p_vals = [], []; final_p_use_rsi_f_vals, final_p_rsi_p_vals = [], []
        final_p_rsi_l_vals, final_p_rsi_s_vals = [], []
        final_idx_h1_ema_s, final_idx_h1_ema_l, final_idx_h4_ema_h = [], [], []
        final_idx_h1_adx, final_idx_h1_atr_sl, final_idx_h1_atr_trail = [], [], []
        final_idx_h1_vol_sma, final_idx_h1_rsi = [], []
        seen_ids_final = set()

        for combo_values in param_generator:
            combo = dict(zip(param_keys, combo_values))
            if combo['ema_short_h1'] >= combo['ema_long_h1']: continue
            # ... (기존 param_id 생성 로직과 동일) ...
            temp_combo_for_id = combo.copy()
            if temp_combo_for_id['exit_strategy_type'] == 'FixedRR':
                temp_combo_for_id['trailing_atr_period'] = current_param_ranges['trailing_atr_period'][0] if current_param_ranges['trailing_atr_period'] else 14
                temp_combo_for_id['trailing_atr_multiplier'] = current_param_ranges['trailing_atr_multiplier'][0] if current_param_ranges['trailing_atr_multiplier'] else 1.5
            elif temp_combo_for_id['exit_strategy_type'] == 'TrailingATR':
                temp_combo_for_id['risk_reward_ratio'] = current_param_ranges['risk_reward_ratio'][0] if current_param_ranges['risk_reward_ratio'] else 2.0
            if not temp_combo_for_id.get('use_htf_ema_filter', False): temp_combo_for_id['ema_htf'] = current_param_ranges['ema_htf'][0] if current_param_ranges['ema_htf'] else 50
            if not temp_combo_for_id.get('use_adx_filter', False):
                temp_combo_for_id['adx_period'] = current_param_ranges['adx_period'][0] if current_param_ranges['adx_period'] else 14
                temp_combo_for_id['adx_threshold'] = current_param_ranges['adx_threshold'][0] if current_param_ranges['adx_threshold'] else 20
            if not temp_combo_for_id.get('use_volume_filter', False): temp_combo_for_id['volume_sma_period'] = current_param_ranges['volume_sma_period'][0] if current_param_ranges['volume_sma_period'] else 20
            if not temp_combo_for_id.get('use_rsi_filter', False): # RSI 필터 사용 안 할 때 기본값 설정
                temp_combo_for_id['rsi_period'] = current_param_ranges['rsi_period'][0] if current_param_ranges['rsi_period'] else 14
                temp_combo_for_id['rsi_threshold_long'] = current_param_ranges['rsi_threshold_long'][0] if current_param_ranges['rsi_threshold_long'] else 50
                temp_combo_for_id['rsi_threshold_short'] = current_param_ranges['rsi_threshold_short'][0] if current_param_ranges['rsi_threshold_short'] else 50


            param_id_parts = []; c_id = temp_combo_for_id
            param_id_parts.append(f"EMA{c_id['ema_short_h1']}-{c_id['ema_long_h1']}-H4_{c_id['ema_htf']}")
            if c_id.get('use_adx_filter', False): param_id_parts.append(f"ADX{c_id['adx_period']}_{c_id['adx_threshold']}")
            else: param_id_parts.append("NoADX")
            param_id_parts.append(f"SL{c_id['atr_multiplier_sl']}ATR{c_id['atr_period_sl']}")
            if c_id['exit_strategy_type'] == 'FixedRR': param_id_parts.append(f"RR{c_id['risk_reward_ratio']}")
            elif c_id['exit_strategy_type'] == 'TrailingATR': param_id_parts.append(f"Trail{c_id['trailing_atr_multiplier']}ATR{c_id['trailing_atr_period']}")
            if c_id.get('use_htf_ema_filter', False): param_id_parts.append("H4Filt")
            else: param_id_parts.append("NoH4Filt")
            if c_id.get('use_volume_filter', False): param_id_parts.append(f"VolFilt{c_id['volume_sma_period']}")
            else: param_id_parts.append("NoVolFilt")
            if c_id.get('use_rsi_filter', False): param_id_parts.append(f"RSI{c_id['rsi_period']}_{c_id['rsi_threshold_long']}-{c_id['rsi_threshold_short']}")
            else: param_id_parts.append("NoRSIFilt") # RSI 필터 사용 안 함 명시
            param_id_parts.append(f"Risk{c_id['risk_per_trade_percentage']*100:.1f}%"); param_id_str = "_".join(map(str, param_id_parts))


            if param_id_str not in seen_ids_final:
                seen_ids_final.add(param_id_str); final_param_combos_dicts.append(combo); final_param_ids.append(param_id_str)
                # ... (기존 파라미터 값 및 인덱스 리스트 append 로직과 동일) ...
                final_p_ema_s_vals.append(combo['ema_short_h1']); final_p_ema_l_vals.append(combo['ema_long_h1']); final_p_ema_h_vals.append(combo['ema_htf'])
                final_p_adx_p_vals.append(combo['adx_period']); final_p_adx_t_vals.append(combo['adx_threshold']); final_p_atr_p_sl_vals.append(combo['atr_period_sl']); final_p_atr_m_sl_vals.append(combo['atr_multiplier_sl'])
                final_p_rr_vals.append(combo['risk_reward_ratio']); final_p_use_htf_f_vals.append(1 if combo.get('use_htf_ema_filter') else 0); final_p_use_adx_f_vals.append(1 if combo.get('use_adx_filter') else 0)
                final_p_risk_vals.append(combo['risk_per_trade_percentage']); final_p_exit_type_vals.append(EXIT_STRATEGY_FIXED_RR_GPU if combo['exit_strategy_type'] == 'FixedRR' else EXIT_STRATEGY_TRAILING_ATR_GPU)
                final_p_trail_atr_p_vals.append(combo['trailing_atr_period']); final_p_trail_atr_m_vals.append(combo['trailing_atr_multiplier'])
                final_p_use_vol_f_vals.append(1 if combo.get('use_volume_filter') else 0); final_p_vol_sma_p_vals.append(combo['volume_sma_period']); final_p_use_rsi_f_vals.append(1 if combo.get('use_rsi_filter') else 0); final_p_rsi_p_vals.append(combo['rsi_period'])
                final_p_rsi_l_vals.append(combo['rsi_threshold_long']); final_p_rsi_s_vals.append(combo['rsi_threshold_short'])
                final_idx_h1_ema_s.append(common_indicator_map.get(('ema', combo['ema_short_h1']), -1)); final_idx_h1_ema_l.append(common_indicator_map.get(('ema', combo['ema_long_h1']), -1)); final_idx_h4_ema_h.append(common_indicator_map.get(('ema', combo['ema_htf']), -1) if combo.get('use_htf_ema_filter') else -1)
                final_idx_h1_adx.append(common_indicator_map.get(('adx', combo['adx_period']), -1) if combo.get('use_adx_filter') else -1); final_idx_h1_atr_sl.append(common_indicator_map.get(('atr', combo['atr_period_sl']), -1)); final_idx_h1_atr_trail.append(common_indicator_map.get(('atr', combo['trailing_atr_period']), -1) if combo['exit_strategy_type'] == 'TrailingATR' else -1)
                final_idx_h1_vol_sma.append(common_indicator_map.get(('vol_sma', combo['volume_sma_period']), -1) if combo.get('use_volume_filter') else -1); final_idx_h1_rsi.append(common_indicator_map.get(('rsi', combo['rsi_period']), -1) if combo.get('use_rsi_filter') else -1)


        num_final_combinations = len(final_param_ids)
        if num_final_combinations == 0: logger.warning(f"{is_period_label}: 유효한 파라미터 조합 없음."); return None, None
        logger.info(f"{is_period_label}: 총 {num_final_combinations}개 고유 파라미터 조합으로 최적화 진행.")

        is_all_results_summary = []
        num_is_batches = math.ceil(num_final_combinations / BATCH_SIZE) if BATCH_SIZE > 0 else 1

        for batch_num in range(num_is_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min((batch_num + 1) * BATCH_SIZE, num_final_combinations)
            current_batch_size = end_idx - start_idx
            if current_batch_size <=0: continue
            
            # ... (기존 배치별 GPU 메모리 할당 및 데이터 전송 로직과 동일) ...
            d_p_ema_s_vals_batch=cuda.to_device(np.array(final_p_ema_s_vals[start_idx:end_idx],dtype=np.int32))
            d_p_ema_l_vals_batch=cuda.to_device(np.array(final_p_ema_l_vals[start_idx:end_idx],dtype=np.int32))
            # ... (나머지 파라미터들도 동일하게 배치 처리)
            d_p_ema_h_vals_batch=cuda.to_device(np.array(final_p_ema_h_vals[start_idx:end_idx],dtype=np.int32))
            d_p_adx_p_vals_batch=cuda.to_device(np.array(final_p_adx_p_vals[start_idx:end_idx],dtype=np.int32))
            d_p_adx_t_vals_batch=cuda.to_device(np.array(final_p_adx_t_vals[start_idx:end_idx],dtype=np.float64))
            d_p_atr_p_sl_vals_batch=cuda.to_device(np.array(final_p_atr_p_sl_vals[start_idx:end_idx],dtype=np.int32))
            d_p_atr_m_sl_vals_batch=cuda.to_device(np.array(final_p_atr_m_sl_vals[start_idx:end_idx],dtype=np.float64))
            d_p_rr_vals_batch=cuda.to_device(np.array(final_p_rr_vals[start_idx:end_idx],dtype=np.float64))
            d_p_use_htf_f_vals_batch=cuda.to_device(np.array(final_p_use_htf_f_vals[start_idx:end_idx],dtype=np.int8))
            d_p_use_adx_f_vals_batch=cuda.to_device(np.array(final_p_use_adx_f_vals[start_idx:end_idx],dtype=np.int8))
            d_p_risk_vals_batch=cuda.to_device(np.array(final_p_risk_vals[start_idx:end_idx],dtype=np.float64))
            d_p_exit_type_vals_batch=cuda.to_device(np.array(final_p_exit_type_vals[start_idx:end_idx],dtype=np.int8))
            d_p_trail_atr_p_vals_batch=cuda.to_device(np.array(final_p_trail_atr_p_vals[start_idx:end_idx],dtype=np.int32))
            d_p_trail_atr_m_vals_batch=cuda.to_device(np.array(final_p_trail_atr_m_vals[start_idx:end_idx],dtype=np.float64))
            d_p_use_vol_f_vals_batch=cuda.to_device(np.array(final_p_use_vol_f_vals[start_idx:end_idx],dtype=np.int8))
            d_p_vol_sma_p_vals_batch=cuda.to_device(np.array(final_p_vol_sma_p_vals[start_idx:end_idx],dtype=np.int32))
            d_p_use_rsi_f_vals_batch=cuda.to_device(np.array(final_p_use_rsi_f_vals[start_idx:end_idx],dtype=np.int8))
            d_p_rsi_p_vals_batch=cuda.to_device(np.array(final_p_rsi_p_vals[start_idx:end_idx],dtype=np.int32))
            d_p_rsi_l_vals_batch=cuda.to_device(np.array(final_p_rsi_l_vals[start_idx:end_idx],dtype=np.float64))
            d_p_rsi_s_vals_batch=cuda.to_device(np.array(final_p_rsi_s_vals[start_idx:end_idx],dtype=np.float64))
            d_idx_h1_ema_s_batch=cuda.to_device(np.array(final_idx_h1_ema_s[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_ema_l_batch=cuda.to_device(np.array(final_idx_h1_ema_l[start_idx:end_idx],dtype=np.int32))
            d_idx_h4_ema_h_batch=cuda.to_device(np.array(final_idx_h4_ema_h[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_adx_batch=cuda.to_device(np.array(final_idx_h1_adx[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_atr_sl_batch=cuda.to_device(np.array(final_idx_h1_atr_sl[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_atr_trail_batch=cuda.to_device(np.array(final_idx_h1_atr_trail[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_vol_sma_batch=cuda.to_device(np.array(final_idx_h1_vol_sma[start_idx:end_idx],dtype=np.int32))
            d_idx_h1_rsi_batch=cuda.to_device(np.array(final_idx_h1_rsi[start_idx:end_idx],dtype=np.int32))

            d_out_final_balances_batch = cuda.device_array(current_batch_size, dtype=np.float64)
            d_out_total_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32)
            d_out_win_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32)
            d_out_error_flags_batch = cuda.device_array(current_batch_size, dtype=np.int8)
            d_out_pnl_percentages_batch = cuda.device_array(current_batch_size, dtype=np.float64)
            d_out_profit_factors_batch = cuda.device_array(current_batch_size, dtype=np.float64)
            d_out_max_drawdowns_batch = cuda.device_array(current_batch_size, dtype=np.float64)


            run_batch_backtest_gpu_kernel[current_batch_size, 1](
                d_is_open_p, d_is_high_p, d_is_low_p, d_is_close_p, d_is_volume,
                d_is_h1_indicators, d_is_h4_indicators,
                d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch, d_p_adx_p_vals_batch, d_p_adx_t_vals_batch,
                d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch, d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch,
                d_p_risk_vals_batch, d_p_exit_type_vals_batch, d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch,
                d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch, d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch,
                d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch,
                d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch, d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch,
                d_idx_h1_atr_trail_batch, d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch,
                d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch,
                d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch,
                is_data_length, initial_balance, commission_rate_backtest, slippage_rate_per_trade,
                min_trade_size_btc, quantity_precision_bt
            )
            cuda.synchronize()

            h_final_balances_batch = d_out_final_balances_batch.copy_to_host()
            h_total_trades_batch = d_out_total_trades_batch.copy_to_host()
            # ... (기존 결과값 host로 복사 로직과 동일) ...
            h_win_trades_batch = d_out_win_trades_batch.copy_to_host()
            h_error_flags_batch = d_out_error_flags_batch.copy_to_host()
            h_pnl_percentages_batch = d_out_pnl_percentages_batch.copy_to_host()
            h_profit_factors_batch = d_out_profit_factors_batch.copy_to_host()
            h_max_drawdowns_batch = d_out_max_drawdowns_batch.copy_to_host()


            for j in range(current_batch_size):
                global_idx = start_idx + j
                param_id = final_param_ids[global_idx]
                num_trades = h_total_trades_batch[j]
                # JSON 저장용 performance_summary 생성 시 필요한 모든 값 포함
                performance_summary_for_json = {
                    "param_id": param_id,
                    "initial_balance": initial_balance,
                    "final_balance": round(h_final_balances_batch[j], 2),
                    "total_net_pnl": round(h_final_balances_batch[j] - initial_balance, 2),
                    "total_net_pnl_percentage": round(h_pnl_percentages_batch[j], 2),
                    "num_trades": int(num_trades), # int로 변환
                    "num_wins": int(h_win_trades_batch[j]), # int로 변환
                    "num_losses": int(num_trades - h_win_trades_batch[j]), # int로 변환
                    "win_rate_percentage": round((h_win_trades_batch[j] / num_trades) * 100 if num_trades > 0 else 0, 2),
                    "profit_factor": round(h_profit_factors_batch[j], 2) if h_profit_factors_batch[j] != float('inf') else 'inf', # inf는 문자열로
                    "max_drawdown_percentage": round(h_max_drawdowns_batch[j], 2),
                    "error": bool(h_error_flags_batch[j]) # bool로 변환
                }
                # 기존 is_all_results_summary에는 original_params도 포함
                full_performance_summary = performance_summary_for_json.copy()
                full_performance_summary["original_params"] = final_param_combos_dicts[global_idx]
                is_all_results_summary.append(full_performance_summary)
            
            # 배치 GPU 메모리 명시적 해제
            del d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch, d_p_adx_p_vals_batch, d_p_adx_t_vals_batch
            del d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch, d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch
            del d_p_risk_vals_batch, d_p_exit_type_vals_batch, d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch
            del d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch, d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch
            del d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch
            del d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch, d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch
            del d_idx_h1_atr_trail_batch, d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch
            del d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch
            del d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch
            cp.get_default_memory_pool().free_all_blocks()


        best_is_performer = None; best_is_metric_val = -float('inf')
        for res in is_all_results_summary:
            if not res["error"] and res['num_trades'] >= min_trades_for_best_param:
                metric = res.get('total_net_pnl_percentage', -float('inf'))
                if isinstance(metric, (int, float)) and metric > best_is_metric_val:
                    best_is_metric_val = metric; best_is_performer = res
        
        if best_is_performer:
            logger.info(f"{is_period_label}: 최고 성과 파라미터 ID: {best_is_performer['param_id']}, PnL: {best_is_performer['total_net_pnl_percentage']}%")
            print_performance_report(best_is_performer, best_is_performer['param_id'], context=f"{is_period_label} IS Best")
            
            # <<< JSON 파일 저장 호출 위치 >>>
            # best_is_performer에는 'original_params' (딕셔너리)와 'param_id' 및 기타 성과 지표가 포함되어 있음
            # performance_summary_for_json 형식으로 전달
            is_performance_summary_for_json = {k: v for k, v in best_is_performer.items() if k != 'original_params'}

            saved_json_path = save_optimized_params_to_json(
                best_params_dict=best_is_performer['original_params'], # 최적 파라미터 딕셔너리
                performance_summary=is_performance_summary_for_json,  # IS 성과 요약
                wfa_info=wfa_current_info, # WFA 윈도우 정보
                output_dir=CLOUD_SYNC_DIR_WFA_RESULTS
            )
            if saved_json_path:
                logger.info(f"{is_period_label}: 최적 파라미터 JSON 파일 저장 성공: {saved_json_path}")
            else:
                logger.error(f"{is_period_label}: 최적 파라미터 JSON 파일 저장 실패.")

            return best_is_performer['original_params'], best_is_performer # 기존 반환값 유지
        else:
            logger.warning(f"{is_period_label}: 유효한 최고 성과 파라미터를 찾지 못함.")
            return None, None
    finally:
        del d_is_open_p, d_is_high_p, d_is_low_p, d_is_close_p, d_is_volume, d_is_h1_indicators, d_is_h4_indicators
        cp.get_default_memory_pool().free_all_blocks()


def run_single_test_for_oos_period(
    oos_df_primary, oos_df_htf,
    best_params_from_is,
    common_indicator_map, common_unique_configs, common_num_total_series,
    oos_period_label
):
    # ... (기존 OOS 테스트 로직과 동일) ...
    logger.info(f"--- {oos_period_label}: 아웃-오브-샘플 테스트 시작 (파라미터: {best_params_from_is}) ---")
    oos_data_length = len(oos_df_primary)
    if oos_data_length == 0:
        logger.warning(f"{oos_period_label}: 아웃-오브-샘플 데이터 없음. 테스트 스킵.")
        return None

    oos_master_h1_indicators_np = precompute_all_indicators_for_gpu(oos_df_primary, common_unique_configs, common_indicator_map, common_num_total_series, f"{oos_period_label}_H1", oos_data_length)
    oos_master_h4_indicators_np = None
    if oos_df_htf is not None and not oos_df_htf.empty:
        oos_master_h4_indicators_np = precompute_all_indicators_for_gpu(oos_df_htf, common_unique_configs, common_indicator_map, common_num_total_series, f"{oos_period_label}_H4", oos_data_length)
    else:
        oos_master_h4_indicators_np = np.full_like(oos_master_h1_indicators_np, np.nan) if oos_master_h1_indicators_np is not None else None

    if oos_master_h1_indicators_np is None: logger.error(f"{oos_period_label}: H1 마스터 지표 생성 실패."); return None
    
    try:
        oos_open_p_np = oos_df_primary['Open'].to_numpy(dtype=np.float64); d_oos_open_p=cuda.to_device(oos_open_p_np)
        oos_high_p_np = oos_df_primary['High'].to_numpy(dtype=np.float64); d_oos_high_p=cuda.to_device(oos_high_p_np)
        oos_low_p_np = oos_df_primary['Low'].to_numpy(dtype=np.float64); d_oos_low_p=cuda.to_device(oos_low_p_np)
        oos_close_p_np = oos_df_primary['Close'].to_numpy(dtype=np.float64); d_oos_close_p=cuda.to_device(oos_close_p_np)
        oos_volume_np = oos_df_primary['Volume'].to_numpy(dtype=np.float64); d_oos_volume=cuda.to_device(oos_volume_np)

        d_oos_h1_indicators=cuda.to_device(oos_master_h1_indicators_np)
        d_oos_h4_indicators=cuda.to_device(oos_master_h4_indicators_np if oos_master_h4_indicators_np is not None else d_oos_h1_indicators)

        param_dict = best_params_from_is
        d_p_ema_s_vals_oos = cuda.to_device(np.array([param_dict['ema_short_h1']], dtype=np.int32))
        # ... (기존 OOS 파라미터 GPU 전송 로직과 동일) ...
        d_p_ema_l_vals_oos = cuda.to_device(np.array([param_dict['ema_long_h1']], dtype=np.int32))
        d_p_ema_h_vals_oos = cuda.to_device(np.array([param_dict['ema_htf']], dtype=np.int32))
        d_p_adx_p_vals_oos = cuda.to_device(np.array([param_dict['adx_period']], dtype=np.int32))
        d_p_adx_t_vals_oos = cuda.to_device(np.array([param_dict['adx_threshold']], dtype=np.float64))
        d_p_atr_p_sl_vals_oos = cuda.to_device(np.array([param_dict['atr_period_sl']], dtype=np.int32))
        d_p_atr_m_sl_vals_oos = cuda.to_device(np.array([param_dict['atr_multiplier_sl']], dtype=np.float64))
        d_p_rr_vals_oos = cuda.to_device(np.array([param_dict['risk_reward_ratio']], dtype=np.float64))
        d_p_use_htf_f_vals_oos = cuda.to_device(np.array([1 if param_dict.get('use_htf_ema_filter') else 0], dtype=np.int8))
        d_p_use_adx_f_vals_oos = cuda.to_device(np.array([1 if param_dict.get('use_adx_filter') else 0], dtype=np.int8))
        d_p_risk_vals_oos = cuda.to_device(np.array([param_dict['risk_per_trade_percentage']], dtype=np.float64))
        d_p_exit_type_vals_oos = cuda.to_device(np.array([EXIT_STRATEGY_FIXED_RR_GPU if param_dict['exit_strategy_type'] == 'FixedRR' else EXIT_STRATEGY_TRAILING_ATR_GPU], dtype=np.int8))
        d_p_trail_atr_p_vals_oos = cuda.to_device(np.array([param_dict['trailing_atr_period']], dtype=np.int32))
        d_p_trail_atr_m_vals_oos = cuda.to_device(np.array([param_dict['trailing_atr_multiplier']], dtype=np.float64))
        d_p_use_vol_f_vals_oos = cuda.to_device(np.array([1 if param_dict.get('use_volume_filter') else 0], dtype=np.int8))
        d_p_vol_sma_p_vals_oos = cuda.to_device(np.array([param_dict['volume_sma_period']], dtype=np.int32))
        d_p_use_rsi_f_vals_oos = cuda.to_device(np.array([1 if param_dict.get('use_rsi_filter') else 0], dtype=np.int8))
        d_p_rsi_p_vals_oos = cuda.to_device(np.array([param_dict['rsi_period']], dtype=np.int32))
        d_p_rsi_l_vals_oos = cuda.to_device(np.array([param_dict['rsi_threshold_long']], dtype=np.float64))
        d_p_rsi_s_vals_oos = cuda.to_device(np.array([param_dict['rsi_threshold_short']], dtype=np.float64))

        d_idx_h1_ema_s_oos = cuda.to_device(np.array([common_indicator_map.get(('ema', param_dict['ema_short_h1']), -1)], dtype=np.int32))
        # ... (기존 OOS 지표 인덱스 GPU 전송 로직과 동일) ...
        d_idx_h1_ema_l_oos = cuda.to_device(np.array([common_indicator_map.get(('ema', param_dict['ema_long_h1']), -1)], dtype=np.int32))
        d_idx_h4_ema_h_oos = cuda.to_device(np.array([common_indicator_map.get(('ema', param_dict['ema_htf']), -1) if param_dict.get('use_htf_ema_filter') else -1], dtype=np.int32))
        d_idx_h1_adx_oos = cuda.to_device(np.array([common_indicator_map.get(('adx', param_dict['adx_period']), -1) if param_dict.get('use_adx_filter') else -1], dtype=np.int32))
        d_idx_h1_atr_sl_oos = cuda.to_device(np.array([common_indicator_map.get(('atr', param_dict['atr_period_sl']), -1)], dtype=np.int32))
        d_idx_h1_atr_trail_oos = cuda.to_device(np.array([common_indicator_map.get(('atr', param_dict['trailing_atr_period']), -1) if param_dict['exit_strategy_type'] == 'TrailingATR' else -1], dtype=np.int32))
        d_idx_h1_vol_sma_oos = cuda.to_device(np.array([common_indicator_map.get(('vol_sma', param_dict['volume_sma_period']), -1) if param_dict.get('use_volume_filter') else -1], dtype=np.int32))
        d_idx_h1_rsi_oos = cuda.to_device(np.array([common_indicator_map.get(('rsi', param_dict['rsi_period']), -1) if param_dict.get('use_rsi_filter') else -1], dtype=np.int32))


        d_out_final_balances_oos = cuda.device_array(1, dtype=np.float64)
        # ... (기존 OOS 결과 배열 GPU 할당 로직과 동일) ...
        d_out_total_trades_oos = cuda.device_array(1, dtype=np.int32)
        d_out_win_trades_oos = cuda.device_array(1, dtype=np.int32)
        d_out_error_flags_oos = cuda.device_array(1, dtype=np.int8)
        d_out_pnl_percentages_oos = cuda.device_array(1, dtype=np.float64)
        d_out_profit_factors_oos = cuda.device_array(1, dtype=np.float64)
        d_out_max_drawdowns_oos = cuda.device_array(1, dtype=np.float64)


        run_batch_backtest_gpu_kernel[1, 1](
            d_oos_open_p, d_oos_high_p, d_oos_low_p, d_oos_close_p, d_oos_volume,
            d_oos_h1_indicators, d_oos_h4_indicators,
            d_p_ema_s_vals_oos, d_p_ema_l_vals_oos, d_p_ema_h_vals_oos, d_p_adx_p_vals_oos, d_p_adx_t_vals_oos,
            d_p_atr_p_sl_vals_oos, d_p_atr_m_sl_vals_oos, d_p_rr_vals_oos, d_p_use_htf_f_vals_oos, d_p_use_adx_f_vals_oos,
            d_p_risk_vals_oos, d_p_exit_type_vals_oos, d_p_trail_atr_p_vals_oos, d_p_trail_atr_m_vals_oos,
            d_p_use_vol_f_vals_oos, d_p_vol_sma_p_vals_oos, d_p_use_rsi_f_vals_oos, d_p_rsi_p_vals_oos,
            d_p_rsi_l_vals_oos, d_p_rsi_s_vals_oos,
            d_idx_h1_ema_s_oos, d_idx_h1_ema_l_oos, d_idx_h4_ema_h_oos, d_idx_h1_adx_oos, d_idx_h1_atr_sl_oos,
            d_idx_h1_atr_trail_oos, d_idx_h1_vol_sma_oos, d_idx_h1_rsi_oos,
            d_out_final_balances_oos, d_out_total_trades_oos, d_out_win_trades_oos, d_out_error_flags_oos,
            d_out_pnl_percentages_oos, d_out_profit_factors_oos, d_out_max_drawdowns_oos,
            oos_data_length, initial_balance, commission_rate_backtest, slippage_rate_per_trade,
            min_trade_size_btc, quantity_precision_bt
        )
        cuda.synchronize()

        h_final_balance = d_out_final_balances_oos.copy_to_host()[0]
        # ... (기존 OOS 결과 host로 복사 및 performance 객체 생성 로직과 동일) ...
        h_total_trades = d_out_total_trades_oos.copy_to_host()[0]
        h_win_trades = d_out_win_trades_oos.copy_to_host()[0]
        h_error_flag = d_out_error_flags_oos.copy_to_host()[0]
        h_pnl_percentage = d_out_pnl_percentages_oos.copy_to_host()[0]
        h_profit_factor = d_out_profit_factors_oos.copy_to_host()[0]
        h_max_drawdown = d_out_max_drawdowns_oos.copy_to_host()[0]

        param_id_oos = f"OOS_{oos_period_label}_IS_Optimized" # param_id에 oos_period_label 포함
        oos_performance = {
            "param_id": param_id_oos, "original_params": best_params_from_is,
            "initial_balance": initial_balance, "final_balance": round(h_final_balance, 2),
            "total_net_pnl": round(h_final_balance - initial_balance, 2),
            "total_net_pnl_percentage": round(h_pnl_percentage, 2),
            "num_trades": int(h_total_trades), "num_wins": int(h_win_trades),
            "num_losses": int(h_total_trades - h_win_trades),
            "win_rate_percentage": round((h_win_trades / h_total_trades) * 100 if h_total_trades > 0 else 0, 2),
            "profit_factor": round(h_profit_factor, 2) if h_profit_factor != float('inf') else 'inf',
            "max_drawdown_percentage": round(h_max_drawdown, 2), "error": bool(h_error_flag)
        }
        print_performance_report(oos_performance, param_id_oos, context=f"{oos_period_label} OOS Test")
        return oos_performance
    finally:
        # OOS 기간용 GPU 메모리 해제
        del d_oos_open_p, d_oos_high_p, d_oos_low_p, d_oos_close_p, d_oos_volume
        del d_oos_h1_indicators, d_oos_h4_indicators
        del d_p_ema_s_vals_oos, d_p_ema_l_vals_oos, d_p_ema_h_vals_oos, d_p_adx_p_vals_oos, d_p_adx_t_vals_oos
        del d_p_atr_p_sl_vals_oos, d_p_atr_m_sl_vals_oos, d_p_rr_vals_oos, d_p_use_htf_f_vals_oos, d_p_use_adx_f_vals_oos
        del d_p_risk_vals_oos, d_p_exit_type_vals_oos, d_p_trail_atr_p_vals_oos, d_p_trail_atr_m_vals_oos
        del d_p_use_vol_f_vals_oos, d_p_vol_sma_p_vals_oos, d_p_use_rsi_f_vals_oos, d_p_rsi_p_vals_oos
        del d_p_rsi_l_vals_oos, d_p_rsi_s_vals_oos
        del d_idx_h1_ema_s_oos, d_idx_h1_ema_l_oos, d_idx_h4_ema_h_oos, d_idx_h1_adx_oos, d_idx_h1_atr_sl_oos
        del d_idx_h1_atr_trail_oos, d_idx_h1_vol_sma_oos, d_idx_h1_rsi_oos
        del d_out_final_balances_oos, d_out_total_trades_oos, d_out_win_trades_oos, d_out_error_flags_oos
        del d_out_pnl_percentages_oos, d_out_profit_factors_oos, d_out_max_drawdowns_oos
        cp.get_default_memory_pool().free_all_blocks()


# --- Main WFA Execution Logic ---
if __name__ == "__main__":
    logger.info("=== EMA Crossover 전략 워크-포워드 분석기 (WFA_batchii.py V1.2) 실행 시작 ===")
    overall_wfa_start_time = time.time()

    try:
        if not cuda.is_available(): logger.error("CUDA 사용 불가능."); exit()
        if len(cuda.gpus) == 0: logger.error("사용 가능한 GPU 장치 없음."); exit()
        logger.info(f"CUDA 사용 가능. GPU 장치 수: {len(cuda.gpus)}")
        selected_gpu = cuda.get_current_device(); logger.info(f"선택된 GPU: {selected_gpu.name.decode()}")
        cp.cuda.runtime.getDeviceCount(); logger.info("CuPy도 CUDA 장치 인식 완료.")
    except Exception as e: logger.error(f"CUDA/CuPy 초기화 오류: {e}."); exit()

    common_unique_indicator_configs, common_indicator_to_idx_map, common_num_total_indicator_series = get_unique_indicator_params_and_map(param_ranges)

    wfa_start_dt = datetime.strptime(wfa_overall_start_date_str, "%Y-%m-%d")
    wfa_end_dt = datetime.strptime(wfa_overall_end_date_str, "%Y-%m-%d")

    current_is_start_dt = wfa_start_dt
    all_oos_performances = []
    wfa_window_num = 0

    while True:
        wfa_window_num += 1
        logger.info(f"\n\n{'='*20} WFA Window {wfa_window_num} 시작 {'='*20}")

        current_is_end_dt = current_is_start_dt + pd.DateOffset(months=is_period_months) - timedelta(days=1)
        current_oos_start_dt = current_is_end_dt + timedelta(days=1)
        current_oos_end_dt = current_oos_start_dt + pd.DateOffset(months=oos_period_months) - timedelta(days=1)


        if current_is_start_dt > wfa_end_dt or current_is_end_dt > wfa_end_dt :
            logger.info(f"IS 기간({current_is_start_dt.strftime('%Y-%m-%d')}~{current_is_end_dt.strftime('%Y-%m-%d')})이 WFA 종료일({wfa_end_dt.strftime('%Y-%m-%d')})을 초과합니다. WFA 루프 종료.")
            break
        if current_oos_start_dt > wfa_end_dt:
             logger.info(f"OOS 시작일({current_oos_start_dt.strftime('%Y-%m-%d')})이 WFA 종료일({wfa_end_dt.strftime('%Y-%m-%d')}) 이후입니다. 마지막 IS 최적화만 수행 후 WFA 루프 종료.")
             break

        is_start_str = current_is_start_dt.strftime("%Y-%m-%d")
        is_end_str = current_is_end_dt.strftime("%Y-%m-%d")
        oos_start_str = current_oos_start_dt.strftime("%Y-%m-%d")
        actual_oos_end_dt = min(current_oos_end_dt, wfa_end_dt)
        oos_end_str = actual_oos_end_dt.strftime("%Y-%m-%d")

        logger.info(f"Window {wfa_window_num} - IS Period: {is_start_str} to {is_end_str}")
        logger.info(f"Window {wfa_window_num} - OOS Period: {oos_start_str} to {oos_end_str}")

        # JSON 저장을 위한 WFA 윈도우 정보 구성
        wfa_current_run_info = {
            "window_number": wfa_window_num,
            "is_period_start": is_start_str,
            "is_period_end": is_end_str,
            "oos_period_expected_start": oos_start_str,
            "oos_period_expected_end": oos_end_str # 실제 OOS 종료일이 아닌, 계획된 OOS 종료일
        }

        is_df_primary = get_historical_data(symbol_backtest, interval_primary_bt, start_str=is_start_str, end_str=is_end_str)
        if is_df_primary is None or is_df_primary.empty:
            logger.warning(f"Window {wfa_window_num}: IS 기간 데이터 로드 실패. 다음 윈도우로 이동.")
            current_is_start_dt = current_is_start_dt + pd.DateOffset(months=oos_period_months)
            continue
        is_df_htf = None
        if True in param_ranges.get('use_htf_ema_filter', [False]): # HTF 필터 사용 시에만 로드
            # HTF 데이터 로드 시, IS 기간보다 충분히 이전 데이터부터 가져와서 EMA 계산에 필요한 데이터 확보
            # 예: ema_htf 최대값이 100이면, 최소 100봉 + @ 필요
            htf_needed_candles = max(param_ranges.get('ema_htf', [50])) + 50 # 여유분 추가
            # HTF 간격에 따라 필요한 lookback 일수 계산 (대략적으로)
            # 4시간봉 기준: htf_needed_candles * 4시간 / 24시간 = 일수
            days_lookback_htf = math.ceil(htf_needed_candles * (4/24) if interval_htf_bt == Client.KLINE_INTERVAL_4HOUR else htf_needed_candles * (1/24)) # 1시간봉이면 1/24
            
            htf_is_fetch_start_dt = current_is_start_dt - timedelta(days=days_lookback_htf) # lookback 기간 추가 확보
            is_df_htf_raw = get_historical_data(symbol_backtest, interval_htf_bt, start_str=htf_is_fetch_start_dt.strftime("%Y-%m-%d"), end_str=is_end_str)
            if is_df_htf_raw is not None and not is_df_htf_raw.empty:
                is_df_htf_raw.sort_index(inplace=True)
                temp_is_primary_time = pd.DataFrame(index=is_df_primary.index)
                # Timezone 통일 (get_historical_data에서 이미 UTC로 처리됨)
                is_df_htf = pd.merge_asof(temp_is_primary_time.sort_index(), is_df_htf_raw.sort_index(), left_index=True, right_index=True, direction='backward')
                is_df_htf = is_df_htf.reindex(is_df_primary.index, method='ffill') # Primary 인덱스에 맞게 ffill

        best_params_from_is_dict, best_is_performance = run_optimization_for_is_period(
            is_df_primary, is_df_htf, param_ranges,
            common_indicator_to_idx_map, common_unique_indicator_configs, common_num_total_indicator_series,
            f"WFA_Win{wfa_window_num}",
            wfa_current_run_info # JSON 저장용 정보 전달
        )

        if best_params_from_is_dict is None:
            logger.warning(f"Window {wfa_window_num}: IS 기간 최적화 실패 또는 유효 파라미터 없음. 다음 윈도우로 이동.")
            current_is_start_dt = current_is_start_dt + pd.DateOffset(months=oos_period_months)
            continue
        
        if actual_oos_end_dt < current_oos_start_dt :
             logger.info(f"Window {wfa_window_num}: OOS 기간({oos_start_str} ~ {oos_end_str})이 유효하지 않아 OOS 테스트 스킵. 다음 윈도우로 이동.")
             current_is_start_dt = current_is_start_dt + pd.DateOffset(months=oos_period_months)
             continue

        oos_df_primary = get_historical_data(symbol_backtest, interval_primary_bt, start_str=oos_start_str, end_str=oos_end_str)
        if oos_df_primary is None or oos_df_primary.empty:
            logger.warning(f"Window {wfa_window_num}: OOS 기간 데이터 로드 실패. 다음 윈도우로 이동.")
            current_is_start_dt = current_is_start_dt + pd.DateOffset(months=oos_period_months)
            continue
        oos_df_htf = None
        if best_params_from_is_dict.get('use_htf_ema_filter', False): # 최적 파라미터가 HTF 필터 사용 시에만 로드
            htf_needed_candles_oos = max(param_ranges.get('ema_htf', [50])) + 50
            days_lookback_htf_oos = math.ceil(htf_needed_candles_oos * (4/24) if interval_htf_bt == Client.KLINE_INTERVAL_4HOUR else htf_needed_candles_oos * (1/24))
            htf_oos_fetch_start_dt = current_oos_start_dt - timedelta(days=days_lookback_htf_oos)
            oos_df_htf_raw = get_historical_data(symbol_backtest, interval_htf_bt, start_str=htf_oos_fetch_start_dt.strftime("%Y-%m-%d"), end_str=oos_end_str)
            if oos_df_htf_raw is not None and not oos_df_htf_raw.empty:
                oos_df_htf_raw.sort_index(inplace=True)
                temp_oos_primary_time = pd.DataFrame(index=oos_df_primary.index)
                oos_df_htf = pd.merge_asof(temp_oos_primary_time.sort_index(), oos_df_htf_raw.sort_index(), left_index=True, right_index=True, direction='backward')
                oos_df_htf = oos_df_htf.reindex(oos_df_primary.index, method='ffill')


        oos_performance = run_single_test_for_oos_period(
            oos_df_primary, oos_df_htf, best_params_from_is_dict,
            common_indicator_to_idx_map, common_unique_indicator_configs, common_num_total_indicator_series,
            f"WFA_Win{wfa_window_num}"
        )

        if oos_performance:
            oos_performance['wfa_window'] = wfa_window_num
            oos_performance['is_period'] = f"{is_start_str}_to_{is_end_str}"
            oos_performance['oos_period'] = f"{oos_start_str}_to_{oos_end_str}" # 실제 OOS 기간
            oos_performance['best_is_param_id'] = best_is_performance['param_id'] if best_is_performance else "N/A"
            all_oos_performances.append(oos_performance)

        current_is_start_dt = current_is_start_dt + pd.DateOffset(months=oos_period_months) # 다음 IS 기간 시작일 업데이트
        gc.collect() # 메모리 정리

    logger.info("\n\n" + "="*20 + " 전체 워크-포워드 분석 결과 요약 " + "="*20)
    if all_oos_performances:
        oos_summary_df = pd.DataFrame(all_oos_performances)
        logger.info(f"\n총 {len(oos_summary_df)}개 OOS 기간 테스트 완료.")
        cols_to_show = ['wfa_window', 'oos_period', 'best_is_param_id', 'total_net_pnl_percentage', 'num_trades', 'win_rate_percentage', 'profit_factor', 'max_drawdown_percentage']
        logger.info("\n--- OOS 기간별 성과 ---")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            logger.info(f"\n{oos_summary_df[cols_to_show].to_string(index=False)}\n")

        avg_oos_pnl_perc = oos_summary_df['total_net_pnl_percentage'].mean()
        avg_oos_win_rate = oos_summary_df['win_rate_percentage'].mean()
        valid_profit_factors = pd.to_numeric(oos_summary_df['profit_factor'], errors='coerce').dropna()
        avg_oos_profit_factor = valid_profit_factors.mean() if not valid_profit_factors.empty else np.nan
        avg_oos_mdd = oos_summary_df['max_drawdown_percentage'].mean()
        total_oos_trades = oos_summary_df['num_trades'].sum()

        logger.info("\n--- 전체 OOS 기간 평균 성과 ---")
        logger.info(f"평균 OOS PnL %: {avg_oos_pnl_perc:.2f}%")
        logger.info(f"평균 OOS 승률: {avg_oos_win_rate:.2f}%")
        logger.info(f"평균 OOS Profit Factor: {avg_oos_profit_factor:.2f}")
        logger.info(f"평균 OOS MDD %: {avg_oos_mdd:.2f}%")
        logger.info(f"총 OOS 거래 수: {total_oos_trades}")
        
        wfa_equity_curve = [initial_balance]
        current_equity = initial_balance
        for index, row in oos_summary_df.iterrows():
            # 각 OOS 기간의 실제 PnL 금액을 사용 (final_balance - initial_balance)
            # 이 방식은 각 OOS 기간이 독립적인 initial_balance에서 시작한다고 가정하고 누적함
            # 좀 더 정확한 자본 곡선은 각 OOS 기간의 실제 거래 내역을 이어 붙여야 함
            # 여기서는 각 OOS 기간의 성과를 바탕으로 단순 누적하여 추세를 봄
            period_pnl_amount = row['total_net_pnl'] # total_net_pnl은 해당 OOS 기간만의 PnL
            current_equity += period_pnl_amount # 이전 자본에 PnL을 더함
            wfa_equity_curve.append(current_equity)
        
        if len(wfa_equity_curve) > 1: # 초기 자본 외에 데이터가 있다면
            overall_wfa_pnl_percentage = (wfa_equity_curve[-1] - initial_balance) / initial_balance * 100
            logger.info(f"워크-포워드 전체 PnL % (OOS 기간 PnL 금액 누적): {overall_wfa_pnl_percentage:.2f}%")
            logger.info(f"워크-포워드 최종 자본 (OOS 기간 PnL 금액 누적): {wfa_equity_curve[-1]:.2f}")

        else:
            logger.info("OOS 기간 성과가 없어 전체 PnL%를 계산할 수 없습니다.")
    else:
        logger.info("OOS 기간 테스트 결과가 없습니다.")

    overall_wfa_duration = time.time() - overall_wfa_start_time
    logger.info(f"총 WFA 실행 시간: {timedelta(seconds=overall_wfa_duration)}")
    logger.info("=== EMA Crossover 전략 워크-포워드 분석기 실행 완료 ===")
