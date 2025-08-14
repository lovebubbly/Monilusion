# -*- coding: utf-8 -*-
# EMA Crossover Strategy Backtester (V5.11 - Optimized Parameter Ranges for Fine-tuning)
# pip install python-binance pandas numpy python-dotenv pandas_ta openpyxl numba cupy-cuda12x (or your cuda version)

import os
import time
from dotenv import load_dotenv
from binance.client import Client, BinanceAPIException
import pandas as pd
import numpy as np
import pandas_ta as ta # pandas_ta 임포트 확인
import math
import logging
from datetime import datetime, timedelta
import itertools
from numba import cuda
import cupy as cp
import gc
# from multiprocessing import Pool, cpu_count # 더 이상 사용 안 함
from functools import partial # 더 이상 사용 안 함
import math # math 추가 (ceil 사용)
import json # JSON 저장을 위해 추가

# --- 로깅 설정 ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(filename)s:%(lineno)d - %(message)s')
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler) # 기존 핸들러 제거
logger.addHandler(log_handler_stream)

# --- .env 파일 로드 ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env 파일 로드 완료.")
else:
    logger.warning("경고: .env 파일 없음.")

# --- 오프라인 모드 설정 (로컬 OHLCV 로딩) ---
OFFLINE_OHLCV_H1 = os.getenv('OFFLINE_OHLCV_H1')  # CSV/Parquet 경로
OFFLINE_OHLCV_H4 = os.getenv('OFFLINE_OHLCV_H4')  # CSV/Parquet 경로 (선택)
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
interval_primary_bt = Client.KLINE_INTERVAL_1HOUR
interval_htf_bt = Client.KLINE_INTERVAL_4HOUR
num_klines_to_fetch = None # None으로 설정 시 start_date_str부터 end_date_str까지 모든 데이터 로드
start_date_str = "2023-01-01" # 분석 문서와 동일한 기간 시작
end_date_str = "2025-05-01"   # 분석 문서와 동일한 기간 종료
initial_balance = 10000
commission_rate_backtest = 0.0005 # 거래 수수료 (taker 기준 0.04%이나, maker/taker 혼용 및 슬리피지 고려하여 0.05%)
slippage_rate_per_trade = 0.0002 # 거래당 슬리피지 (0.02%)
min_trade_size_btc = 0.001 # BTC 최소 거래량
price_precision_bt = 2 # BTCUSDT 가격 소수점 자릿수
quantity_precision_bt = 3 # BTCUSDT 수량 소수점 자릿수

# --- 테스트할 파라미터 범위 정의 (V5.11 - 최고 성과 전략 기반 최적화) ---
# 기본 파라미터 스윕 범위(최근 개선안 중심). 프로필에 의해 아래에서 덮어쓸 수 있음.
param_ranges = {
    # 추세/모멘텀: 노이즈 회피를 위해 EMA 간격 확대
    'ema_short_h1': [10, 12, 14],
    'ema_long_h1': [50, 100],

    # HTF 필터 고정 사용, 기간 후보 2개
    'use_htf_ema_filter': [True],
    'ema_htf': [50, 100],

    # ADX 필터 강화
    'use_adx_filter': [True],
    'adx_period': [14],
    'adx_threshold': [25, 30],

    # 손절(ATR) 완화: 과도한 SL 확대를 줄임
    'atr_period_sl': [14, 21],
    'atr_multiplier_sl': [2.2, 2.6],

    # 익절/트레일 두 전략 비교
    'exit_strategy_type': ['TrailingATR', 'FixedRR'],
    'risk_reward_ratio': [2.0, 2.5, 3.0],
    'trailing_atr_period': [14, 21],
    'trailing_atr_multiplier': [1.5, 2.0],

    # 볼륨/RSI 필터 유지
    'use_volume_filter': [True],
    'volume_sma_period': [20, 30],
    'use_rsi_filter': [True],
    'rsi_period': [21],
    'rsi_threshold_long': [50, 52],
    'rsi_threshold_short': [45, 47],

    # 리스크 노출 축소 범위
    'risk_per_trade_percentage': [0.01, 0.015, 0.02],

    # Phase 1 Quick Wins Parameters (from real_M1.py)
    'adx_threshold_regime': [20.0, 25.0, 30.0],
    'atr_percent_threshold_regime': [1.5, 2.0, 2.5, 3.0],
    'time_stop_period_hours': [24, 48, 72], # 1day, 2day, 3day
    'profit_threshold_for_trail': [0.5, 1.0, 1.5, 2.0], # Multiplier of entry_atr
    'max_consecutive_losses': [3, 4, 5],
    'cooldown_period_bars': [12, 24, 48], # 12h, 24h, 48h

    # Short Position Redesign Parameters
    'adx_threshold_for_short': [25.0, 30.0, 35.0],
    'price_breakdown_period': [3, 5, 7],
    'rsi_momentum_threshold': [40.0, 45.0, 50.0],
}

# --- 프로필/기간 프리셋: 원전략(Baseline) vs 개선안(Improved) 헤드투헤드 실험용 ---
# 환경변수 PROFILE_NAME, PERIOD_PRESET 로도 지정 가능
#   PROFILE_NAME ∈ { 'baseline', 'improved' }
#   PERIOD_PRESET ∈ { 'recent', 'stress_july', 'full' }
# PROFILE_NAME = os.getenv('PROFILE_NAME', 'improved').lower()
PERIOD_PRESET = os.getenv('PERIOD_PRESET', 'full').lower()

PROFILES = {
    # real_M1.py의 기존 기본값에 최대한 맞춘 원전략 단일 조합
    'baseline': {
        'ema_short_h1': [19],
        'ema_long_h1': [20],
        'use_htf_ema_filter': [False],
        'ema_htf': [50],
        'use_adx_filter': [False],
        'adx_period': [14],
        'adx_threshold': [20],
        'atr_period_sl': [12],
        'atr_multiplier_sl': [2.8],
        'exit_strategy_type': ['FixedRR'],
        'risk_reward_ratio': [2.8],
        'trailing_atr_period': [14],
        'trailing_atr_multiplier': [2.0],
        'use_volume_filter': [False],
        'volume_sma_period': [20],
        'use_rsi_filter': [True],
        'rsi_period': [24],
        'rsi_threshold_long': [50],
        'rsi_threshold_short': [47],
        'risk_per_trade_percentage': [0.022],
    },
    # 현재 개선안 단일 조합(안정화 버전)
    'improved': {
        'ema_short_h1': [14],
        'ema_long_h1': [100],
        'use_htf_ema_filter': [True],
        'ema_htf': [50],
        'use_adx_filter': [True],
        'adx_period': [14],
        'adx_threshold': [30],
        'atr_period_sl': [14],
        'atr_multiplier_sl': [2.6],
        'exit_strategy_type': ['FixedRR'],
        'risk_reward_ratio': [2.5],
        'trailing_atr_period': [14],
        'trailing_atr_multiplier': [1.5],
        'use_volume_filter': [True],
        'volume_sma_period': [30],
        'use_rsi_filter': [True],
        'rsi_period': [21],
        'rsi_threshold_long': [50],
        'rsi_threshold_short': [45],
        'risk_per_trade_percentage': [0.015],
        # New parameters from real_M1.py
        'adx_threshold_regime': [25.0],
        'atr_percent_threshold_regime': [2.0],
        'time_stop_period_hours': [48],
        'profit_threshold_for_trail': [1.0],
        'max_consecutive_losses': [4],
        'cooldown_period_bars': [24],
        'adx_threshold_for_short': [25.0],
        'price_breakdown_period': [5],
        'rsi_momentum_threshold': [45.0],
    },
}

PERIOD_PRESETS = {
    # 최근 구간(실전 근접) 비교
    'recent': ("2025-05-01", "2025-08-14"),
    # 스트레스 구간(7월 급락 검증)
    'stress_july': ("2025-07-01", "2025-08-01"),
    # 전체(default)
    'full': (None, None),
}

# 프로필 적용: 상단 기본 스윕 범위를 덮어씀(단일 값 중심)
# --- 프로필 적용: 상단 기본 스윕 범위를 덮어씀(단일 값 중심) ---
# if PROFILE_NAME in PROFILES:
#     param_ranges = PROFILES[PROFILE_NAME]
#     logger.info(f"프로필 적용: {PROFILE_NAME} → 단일 조합/축소 범위로 실행")
# else:
#     logger.info(f"프로필 미지정 또는 알 수 없음(PROFILE_NAME={PROFILE_NAME}). 기본 스윕 범위 사용")

# --- 기간 프리셋 적용: 필요 시 시작/종료일 덮어쓰기 ---
# if PERIOD_PRESET in PERIOD_PRESETS:
#     _s, _e = PERIOD_PRESETS[PERIOD_PRESET]
#     if _s:
#         start_date_str = _s
#     if _e:
#         end_date_str = _e
#     logger.info(f"기간 프리셋 적용: {PERIOD_PRESET} → {start_date_str} ~ {end_date_str}")
# else:
#     logger.info(f"기간 프리셋 미지정 또는 알 수 없음(PERIOD_PRESET={PERIOD_PRESET}). 기본 기간 사용: {start_date_str} ~ {end_date_str}")


# --- GPU 커널 및 백테스팅 관련 상수 ---
BATCH_SIZE = 1000000  # 한 번에 GPU에서 처리할 파라미터 조합 수 (GPU 메모리에 맞게 조절)
NUM_PERFORMANCE_METRICS = 7       # 최종 성과 지표 수

# Numba JIT에서 사용할 상수 (GPU 커널 내에서 직접 사용)
POSITION_NONE_GPU = 0
POSITION_LONG_GPU = 1
POSITION_SHORT_GPU = -1
EXIT_REASON_NONE_GPU = -1
EXIT_REASON_SL_GPU = 0
EXIT_REASON_TP_GPU = 1
EXIT_REASON_TRAIL_SL_GPU = 2
EXIT_REASON_TIME_STOP_GPU = 3
EXIT_STRATEGY_FIXED_RR_GPU = 0
EXIT_STRATEGY_TRAILING_ATR_GPU = 1


def _read_ohlcv_local(path):
    """CSV/Parquet에서 OHLCV를 읽어 cudare.py 내부 포맷으로 반환.
    필요한 컬럼: ['Open time','Open','High','Low','Close','Volume'] 또는 인덱스에 시간+ 위 5컬럼.
    """
    try:
        if path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        elif path.lower().endswith('.parquet') or path.lower().endswith('.pq'):
            df = pd.read_parquet(path)
        else:
            logger.error(f"지원하지 않는 파일 형식: {path}")
            return None
    except Exception as e:
        logger.error(f"로컬 OHLCV 로딩 실패: {e}")
        return None

    # 컬럼 정규화
    cols = {c.lower().strip(): c for c in df.columns}
    def has(col):
        return col in df.columns
    # 인덱스에 시간이 있고 컬럼명이 대문자(OHLCV)인 경우 처리
    if 'Open time' not in df.columns and 'open time' not in cols:
        if df.index.name and 'time' in str(df.index.name).lower():
            df = df.reset_index().rename(columns={df.columns[0]: 'Open time'})
        elif 'timestamp' in cols:
            df.rename(columns={cols['timestamp']: 'Open time'}, inplace=True)
        elif 'date' in cols:
            df.rename(columns={cols['date']: 'Open time'}, inplace=True)
    # 나머지 컬럼명 보정
    rename_map = {}
    for want in ['Open','High','Low','Close','Volume']:
        low = want.lower()
        if want not in df.columns:
            if low in cols:
                rename_map[cols[low]] = want
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required = ['Open time','Open','High','Low','Close','Volume']
    if not all(col in df.columns for col in required):
        logger.error(f"로컬 파일에 필요한 컬럼 없음. 필요: {required}, 현재: {list(df.columns)}")
        return None

    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df[required].copy()
    df.set_index('Open time', inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    logger.info(f"로컬 OHLCV 로드 완료: {path}, {df.index.min()} ~ {df.index.max()}, rows={len(df)}")
    return df


def get_historical_data(symbol, interval, start_str=None, end_str=None, limit=1000, max_klines=None):
    """지정된 기간 또는 최대 Klines 수만큼 과거 데이터를 바이낸스에서 로드합니다."""
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

    if start_time_ms_calc is None and max_klines: # 시작 시간이 없고 max_klines만 있을 때 계산
        try:
            # pandas Timedelta 문자열로 변환
            pd_interval_str = interval.replace('m', 'T') # '1m', '5m' 등
            if 'h' in interval: pd_interval_str = interval.upper() # '1H', '4H'
            elif 'd' in interval: pd_interval_str = interval.upper() # '1D'
            elif 'w' in interval: pd_interval_str = interval.upper() # '1W'
            interval_td = pd.Timedelta(pd_interval_str)
            start_time_dt = datetime.now() - (interval_td * max_klines)
            start_time_ms_calc = int(start_time_dt.timestamp() * 1000)
        except ValueError as e:
            logger.error(f"Interval 형식 오류 '{interval}'를 Timedelta로 변환 실패: {e}"); return None


    if end_time_ms_calc is None: # 종료 시간이 없으면 현재 시간으로 설정
        end_time_ms_calc = int(datetime.now().timestamp() * 1000)

    if start_time_ms_calc is None:
        logger.error("시작 시간 또는 최대 Klines 수 지정 필요."); return None

    current_start_time_ms = start_time_ms_calc
    total_fetched_klines = 0
    klines_list = []

    while True:
        if max_klines and total_fetched_klines >= max_klines: break
        if current_start_time_ms >= end_time_ms_calc: break # end_time 이후 데이터는 가져오지 않음

        fetch_limit = limit
        if max_klines: # max_klines가 설정된 경우, 남은 klines 수만큼만 요청
            fetch_limit = min(limit, max_klines - total_fetched_klines)
        if fetch_limit <= 0: break # 더 가져올 필요가 없으면 종료

        try:
            # futures_klines는 endTime을 포함하지 않으므로, endTime을 넘어서도 마지막 배치는 가져올 수 있도록 함
            klines_batch = client.futures_klines(symbol=symbol, interval=interval, startTime=current_start_time_ms, limit=fetch_limit, endTime=end_time_ms_calc)
            if not klines_batch: break # 더 이상 데이터가 없으면 종료

            klines_list.extend(klines_batch)
            last_kline_close_time = klines_batch[-1][6] # 마지막 봉의 종료 시간
            current_start_time_ms = last_kline_close_time + 1 # 다음 요청 시작 시간 설정
            total_fetched_klines += len(klines_batch)

        except BinanceAPIException as e:
            logger.error(f"API 오류 (데이터 로딩): {e}"); time.sleep(5); # 잠시 후 재시도
        except Exception as e_gen:
            logger.error(f"알 수 없는 오류 (데이터 로딩): {e_gen}"); break
        time.sleep(0.1) # API 요청 제한 회피

    if not klines_list: logger.error("데이터 가져오기 실패."); return None

    df = pd.DataFrame(klines_list, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
    df.drop_duplicates(inplace=True); df.sort_index(inplace=True) # 중복 제거 및 시간순 정렬

    if max_klines and len(df) > max_klines: df = df.iloc[-max_klines:] # 정확히 max_klines 개수만큼 자르기

    logger.info(f"총 {len(df)}개 Klines 로드 완료. 기간: {df.index.min()} ~ {df.index.max()}")
    return df

# def iter_dependent_param_combos(pr):
    # 공통 축 (항상 쓰는 것만)
    common_axes = {
        'ema_short_h1': pr['ema_short_h1'],
        'ema_long_h1':  pr['ema_long_h1'],
        'atr_period_sl': pr['atr_period_sl'],
        'atr_multiplier_sl': pr['atr_multiplier_sl'],
        'risk_per_trade_percentage': pr['risk_per_trade_percentage'],
        # Regime 필터(Phase 1) — 탐색 범위를 유지하고 싶으면 남기고, 고정하고 싶으면 1값만 두기
        'adx_threshold_regime': pr['adx_threshold_regime'],
        'atr_percent_threshold_regime': pr['atr_percent_threshold_regime'],
        'time_stop_period_hours': pr['time_stop_period_hours'],
        'profit_threshold_for_trail': pr['profit_threshold_for_trail'],
        'max_consecutive_losses': pr['max_consecutive_losses'],
        'cooldown_period_bars': pr['cooldown_period_bars'],
        # Short 전용 재료
        'adx_threshold_for_short': pr['adx_threshold_for_short'],
        'price_breakdown_period': pr['price_breakdown_period'],
        'rsi_momentum_threshold': pr['rsi_momentum_threshold'],
    }

    # 조건부 축 묶음
    adx_axes  = (pr['adx_period'], pr['adx_threshold']) if True in pr.get('use_adx_filter', [False]) else [([],[])]
    vol_axes  = (pr['volume_sma_period'],) if True in pr.get('use_volume_filter', [False]) else [([])]
    rsi_axes  = (pr['rsi_period'], pr['rsi_threshold_long'], pr['rsi_threshold_short']) if True in pr.get('use_rsi_filter',[False]) else [([],[],[])]
    htf_axes  = (pr['ema_htf'],) if True in pr.get('use_htf_ema_filter', [False]) else [([])]

    # 공통축 펼치기
    common_keys, common_vals = zip(*common_axes.items())
    for common in product(*common_vals):
        base = dict(zip(common_keys, common))

        # EMA 제약
        for es, el in product(pr['ema_short_h1'], pr['ema_long_h1']):
            if es >= el:  # 불량 조합 제거
                continue

            # 필터 분기
            for adx in product(*adx_axes):
                for vol in product(*vol_axes):
                    for rsi in product(*rsi_axes):
                        for htf in product(*htf_axes):

                            # Exit 전략 분기: 합(+)으로 생성
                            # FixedRR
                            for rr in pr['risk_reward_ratio']:
                                combo = dict(base)
                                combo.update({
                                    'ema_short_h1': es,
                                    'ema_long_h1':  el,
                                    'use_adx_filter': True in pr.get('use_adx_filter', [False]),
                                    'use_volume_filter': True in pr.get('use_volume_filter', [False]),
                                    'use_rsi_filter': True in pr.get('use_rsi_filter', [False]),
                                    'use_htf_ema_filter': True in pr.get('use_htf_ema_filter', [False]),
                                    'exit_strategy_type': 'FixedRR',
                                    'risk_reward_ratio': rr,
                                })
                                # 조건부 파라미터 채우기
                                if combo['use_adx_filter']: 
                                    combo['adx_period'], combo['adx_threshold'] = adx if adx != ([],[]) else (14, 25)
                                else:
                                    combo['adx_period'], combo['adx_threshold'] = pr['adx_period'][0], pr['adx_threshold'][0] # Default if filter is False
                                if combo['use_volume_filter']:
                                    combo['volume_sma_period'] = vol[0] if vol != ([],) else 20
                                else:
                                    combo['volume_sma_period'] = pr['volume_sma_period'][0] # Default if filter is False
                                if combo['use_rsi_filter']:
                                    combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = rsi if rsi != ([],[],[]) else (21,50,45)
                                else:
                                    combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = pr['rsi_period'][0], pr['rsi_threshold_long'][0], pr['rsi_threshold_short'][0] # Default if filter is False
                                if combo['use_htf_ema_filter']:
                                    combo['ema_htf'] = htf[0] if htf != ([],) else 50
                                else:
                                    combo['ema_htf'] = pr['ema_htf'][0] # Default if filter is False
                                yield combo

                            # TrailingATR
                            for t_per in pr['trailing_atr_period']:
                                for t_mul in pr['trailing_atr_multiplier']:
                                    combo = dict(base)
                                    combo.update({
                                        'ema_short_h1': es,
                                        'ema_long_h1':  el,
                                        'use_adx_filter': True in pr.get('use_adx_filter', [False]),
                                        'use_volume_filter': True in pr.get('use_volume_filter', [False]),
                                        'use_rsi_filter': True in pr.get('use_rsi_filter', [False]),
                                        'use_htf_ema_filter': True in pr.get('use_htf_ema_filter', [False]),
                                        'exit_strategy_type': 'TrailingATR',
                                        'trailing_atr_period': t_per,
                                        'trailing_atr_multiplier': t_mul,
                                    })
                                    if combo['use_adx_filter']: 
                                        combo['adx_period'], combo['adx_threshold'] = adx if adx != ([],[]) else (14, 25)
                                    else:
                                        combo['adx_period'], combo['adx_threshold'] = pr['adx_period'][0], pr['adx_threshold'][0] # Default if filter is False
                                    if combo['use_volume_filter']:
                                        combo['volume_sma_period'] = vol[0] if vol != ([],) else 20
                                    else:
                                        combo['volume_sma_period'] = pr['volume_sma_period'][0] # Default if filter is False
                                    if combo['use_rsi_filter']:
                                        combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = rsi if rsi != ([],[],[]) else (21,50,45)
                                    else:
                                        combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = pr['rsi_period'][0], pr['rsi_threshold_long'][0], pr['rsi_threshold_short'][0] # Default if filter is False
                                    if combo['use_htf_ema_filter']:
                                        combo['ema_htf'] = htf[0] if htf != ([],) else 50
                                    else:
                                        combo['ema_htf'] = pr['ema_htf'][0] # Default if filter is False
                                    yield combo

def iter_dependent_param_combos(pr):
    # 공통 축 (항상 쓰는 것만)
    common_axes = {
        'ema_short_h1': pr['ema_short_h1'],
        'ema_long_h1':  pr['ema_long_h1'],
        'atr_period_sl': pr['atr_period_sl'],
        'atr_multiplier_sl': pr['atr_multiplier_sl'],
        'risk_per_trade_percentage': pr['risk_per_trade_percentage'],
        # Regime 필터(Phase 1) — 탐색 범위를 유지하고 싶으면 남기고, 고정하고 싶으면 1값만 두기
        'adx_threshold_regime': pr['adx_threshold_regime'],
        'atr_percent_threshold_regime': pr['atr_percent_threshold_regime'],
        'time_stop_period_hours': pr['time_stop_period_hours'],
        'profit_threshold_for_trail': pr['profit_threshold_for_trail'],
        'max_consecutive_losses': pr['max_consecutive_losses'],
        'cooldown_period_bars': pr['cooldown_period_bars'],
        # Short 전용 재료
        'adx_threshold_for_short': pr['adx_threshold_for_short'],
        'price_breakdown_period': pr['price_breakdown_period'],
        'rsi_momentum_threshold': pr['rsi_momentum_threshold'],
    }

    # 조건부 축 묶음
    adx_axes  = (pr['adx_period'], pr['adx_threshold']) if True in pr.get('use_adx_filter', [False]) else [([],[])]
    vol_axes  = (pr['volume_sma_period'],) if True in pr.get('use_volume_filter', [False]) else [([])]
    rsi_axes  = (pr['rsi_period'], pr['rsi_threshold_long'], pr['rsi_threshold_short']) if True in pr.get('use_rsi_filter',[False]) else [([],[],[])]
    htf_axes  = (pr['ema_htf'],) if True in pr.get('use_htf_ema_filter', [False]) else [([])]

    # 공통축 펼치기
    common_keys, common_vals = zip(*common_axes.items())
    for common in product(*common_vals):
        base = dict(zip(common_keys, common))

        # EMA 제약
        for es, el in product(pr['ema_short_h1'], pr['ema_long_h1']):
            if es >= el:  # 불량 조합 제거
                continue

            # 필터 분기
            for adx in product(*adx_axes):
                for vol in product(*vol_axes):
                    for rsi in product(*rsi_axes):
                        for htf in product(*htf_axes):

                            # Exit 전략 분기: 합(+)으로 생성
                            # FixedRR
                            for rr in pr['risk_reward_ratio']:
                                combo = dict(base)
                                combo.update({
                                    'ema_short_h1': es,
                                    'ema_long_h1':  el,
                                    'use_adx_filter': True in pr.get('use_adx_filter', [False]),
                                    'use_volume_filter': True in pr.get('use_volume_filter', [False]),
                                    'use_rsi_filter': True in pr.get('use_rsi_filter', [False]),
                                    'use_htf_ema_filter': True in pr.get('use_htf_ema_filter', [False]),
                                    'exit_strategy_type': 'FixedRR',
                                    'risk_reward_ratio': rr,
                                })
                                # 조건부 파라미터 채우기
                                if combo['use_adx_filter']: 
                                    combo['adx_period'], combo['adx_threshold'] = adx if adx != ([],[]) else (14, 25)
                                else:
                                    combo['adx_period'], combo['adx_threshold'] = pr['adx_period'][0], pr['adx_threshold'][0] # Default if filter is False
                                if combo['use_volume_filter']:
                                    combo['volume_sma_period'] = vol[0] if vol != ([],) else 20
                                else:
                                    combo['volume_sma_period'] = pr['volume_sma_period'][0] # Default if filter is False
                                if combo['use_rsi_filter']:
                                    combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = rsi if rsi != ([],[],[]) else (21,50,45)
                                else:
                                    combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = pr['rsi_period'][0], pr['rsi_threshold_long'][0], pr['rsi_threshold_short'][0] # Default if filter is False
                                if combo['use_htf_ema_filter']:
                                    combo['ema_htf'] = htf[0] if htf != ([],) else 50
                                else:
                                    combo['ema_htf'] = pr['ema_htf'][0] # Default if filter is False
                                yield combo

                            # TrailingATR
                            for t_per in pr['trailing_atr_period']:
                                for t_mul in pr['trailing_atr_multiplier']:
                                    combo = dict(base)
                                    combo.update({
                                        'ema_short_h1': es,
                                        'ema_long_h1':  el,
                                        'use_adx_filter': True in pr.get('use_adx_filter', [False]),
                                        'use_volume_filter': True in pr.get('use_volume_filter', [False]),
                                        'use_rsi_filter': True in pr.get('use_rsi_filter', [False]),
                                        'use_htf_ema_filter': True in pr.get('use_htf_ema_filter', [False]),
                                        'exit_strategy_type': 'TrailingATR',
                                        'trailing_atr_period': t_per,
                                        'trailing_atr_multiplier': t_mul,
                                    })
                                    if combo['use_adx_filter']: 
                                        combo['adx_period'], combo['adx_threshold'] = adx if adx != ([],[]) else (14, 25)
                                    else:
                                        combo['adx_period'], combo['adx_threshold'] = pr['adx_period'][0], pr['adx_threshold'][0] # Default if filter is False
                                    if combo['use_volume_filter']:
                                        combo['volume_sma_period'] = vol[0] if vol != ([],) else 20
                                    else:
                                        combo['volume_sma_period'] = pr['volume_sma_period'][0] # Default if filter is False
                                    if combo['use_rsi_filter']:
                                        combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = rsi if rsi != ([],[],[]) else (21,50,45)
                                    else:
                                        combo['rsi_period'], combo['rsi_threshold_long'], combo['rsi_threshold_short'] = pr['rsi_period'][0], pr['rsi_threshold_long'][0], pr['rsi_threshold_short'][0] # Default if filter is False
                                    if combo['use_htf_ema_filter']:
                                        combo['ema_htf'] = htf[0] if htf != ([],) else 50
                                    else:
                                        combo['ema_htf'] = pr['ema_htf'][0] # Default if filter is False
                                    yield combo

# --- JSON 직렬화 유틸 (정상 단일 버전) ---
def _json_default(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import cupy as _cp
    except Exception:
        _cp = None

    # NumPy
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()

    # CuPy
    if _cp is not None:
        try:
            if isinstance(obj, _cp.ndarray):
                return _cp.asnumpy(obj).tolist()
        except Exception:
            pass
        try:
            import numbers as _numbers
            if isinstance(obj, _numbers.Number) and obj.__class__.__module__.startswith('cupy'):
                return float(obj)
        except Exception:
            pass

    # pandas/py datetime
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    from datetime import datetime as _dt
    if isinstance(obj, _dt):
        return obj.isoformat()

    if isinstance(obj, (set, tuple)):
        return list(obj)

    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _add_conditional_params(current_combo, param_ranges, conditional_param_groups, group_idx):
    if group_idx == len(conditional_param_groups):
        yield current_combo
        return

    flag_name, controlled_params = conditional_param_groups[group_idx]

    # Case 1: Flag is True
    for flag_val in param_ranges[flag_name]: # Iterate through [True] or [False]
        if flag_val: # If the flag is True, include controlled parameters
            combo_with_flag = current_combo.copy()
            combo_with_flag[flag_name] = True
            
            # Generate combinations for controlled parameters
            controlled_param_values = [param_ranges[p] for p in controlled_params]
            for controlled_combo_values in itertools.product(*controlled_param_values):
                combo_with_controlled = combo_with_flag.copy()
                for i, p_name in enumerate(controlled_params):
                    combo_with_controlled[p_name] = controlled_combo_values[i]
                yield from _add_conditional_params(combo_with_controlled, param_ranges, conditional_param_groups, group_idx + 1)
        else: # If the flag is False, set controlled parameters to default/inactive and move on
            combo_with_flag = current_combo.copy()
            combo_with_flag[flag_name] = False
            for p_name in controlled_params:
                # Set to a default/inactive value. Using the first value from param_ranges as a placeholder.
                # This assumes param_ranges always has at least one value for these.
                combo_with_flag[p_name] = param_ranges[p_name][0] if p_name in param_ranges and param_ranges[p_name] else None 
            yield from _add_conditional_params(combo_with_flag, param_ranges, conditional_param_groups, group_idx + 1)

def generate_valid_param_combinations(param_ranges):
    # Define the structure of conditional parameters and their controlling flags
    conditional_param_groups = [
        ('use_htf_ema_filter', ['ema_htf']),
        ('use_adx_filter', ['adx_period', 'adx_threshold']),
        ('use_volume_filter', ['volume_sma_period']),
        ('use_rsi_filter', ['rsi_period', 'rsi_threshold_long', 'rsi_threshold_short']),
        # Regime filters are always active, so they are not conditional in this sense
        # Short strategy redesign parameters are also always active filters
    ]

    # Extract parameters that are always part of the combination (not controlled by a use_*_filter flag)
    # and not part of the exit strategy branching
    always_active_params = {k: v for k, v in param_ranges.items() if k not in [
        'exit_strategy_type', 'risk_reward_ratio', 'trailing_atr_period', 'trailing_atr_multiplier'
    ] and not any(k in group[1] for group in conditional_param_groups) and k not in [group[0] for group in conditional_param_groups]}

    # Generate combinations for always active parameters
    always_active_keys, always_active_values = zip(*always_active_params.items())
    for always_active_combo_values in itertools.product(*always_active_values):
        base_combo = dict(zip(always_active_keys, always_active_combo_values))

        # Handle exit_strategy_type branching
        for exit_type in param_ranges['exit_strategy_type']:
            current_combo = base_combo.copy()
            current_combo['exit_strategy_type'] = exit_type

            if exit_type == 'FixedRR':
                for rr in param_ranges['risk_reward_ratio']:
                    combo_with_exit = current_combo.copy()
                    combo_with_exit['risk_reward_ratio'] = rr
                    # Set TrailingATR related params to a default/inactive value
                    combo_with_exit['trailing_atr_period'] = param_ranges['trailing_atr_period'][0] if 'trailing_atr_period' in param_ranges else 14
                    combo_with_exit['trailing_atr_multiplier'] = param_ranges['trailing_atr_multiplier'][0] if 'trailing_atr_multiplier' in param_ranges else 1.5
                    
                    # Now, recursively add conditional parameters
                    yield from _add_conditional_params(combo_with_exit, param_ranges, conditional_param_groups, 0)

            elif exit_type == 'TrailingATR':
                for trail_atr_p in param_ranges['trailing_atr_period']:
                    for trail_atr_m in param_ranges['trailing_atr_multiplier']:
                        combo_with_exit = current_combo.copy()
                        combo_with_exit['trailing_atr_period'] = trail_atr_p
                        combo_with_exit['trailing_atr_multiplier'] = trail_atr_m
                        # Set FixedRR related params to a default/inactive value
                        combo_with_exit['risk_reward_ratio'] = param_ranges['risk_reward_ratio'][0] if 'risk_reward_ratio' in param_ranges else 2.0

                        # Now, recursively add conditional parameters
                        yield from _add_conditional_params(combo_with_exit, param_ranges, conditional_param_groups, 0)
    try:
        import numpy as _np  # local import to avoid hard dependency at import time
    except Exception:
        _np = None
    try:
        import cupy as _cp  # optional
    except Exception:
        _cp = None

    # NumPy scalar
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    # NumPy array
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    # CuPy array
    if _cp is not None and isinstance(obj, _cp.ndarray):
        try:
            return _cp.asnumpy(obj).tolist()
        except Exception:
            return str(obj)
    # CuPy number
    if _cp is not None:
        try:
            import numbers as _numbers
            if isinstance(obj, _numbers.Number) and obj.__class__.__module__.startswith('cupy'):
                # e.g., cupy._core.core.scalar
                try:
                    return float(obj)
                except Exception:
                    pass
        except Exception:
            pass
    # pandas/py datetime
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    from datetime import datetime as _dt
    if isinstance(obj, _dt):
        return obj.isoformat()
    # set/tuple
    if isinstance(obj, (set, tuple)):
        return list(obj)
    # Fallback for objects with .item()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    # Last resort string

def get_unique_indicator_params_and_map(param_ranges_dict):
    """파라미터 범위에서 고유한 지표 설정값을 추출하고, 각 지표 시리즈에 대한 인덱스 맵을 생성합니다."""
    unique_params_values = {
        'ema_lengths': sorted(list(set(param_ranges_dict.get('ema_short_h1', []) +
                                      param_ranges_dict.get('ema_long_h1', []) +
                                      param_ranges_dict.get('ema_htf', [])))),
        'adx_periods': sorted(list(set(param_ranges_dict.get('adx_period', [])))),
        'atr_periods': sorted(list(set(param_ranges_dict.get('atr_period_sl', []) +
                                      param_ranges_dict.get('trailing_atr_period', [])))),
        'volume_sma_periods': sorted(list(set(param_ranges_dict.get('volume_sma_period', [])))),
        'rsi_periods': sorted(list(set(param_ranges_dict.get('rsi_period', [])))),
    }

    # 각 파라미터 리스트가 비어있을 경우 기본값 추가 (오류 방지)
    for key in unique_params_values:
        if not unique_params_values[key]:
            if key == 'ema_lengths': unique_params_values[key] = [20] # 예시 기본값
            elif key == 'adx_periods': unique_params_values[key] = [14]
            elif key == 'atr_periods': unique_params_values[key] = [14]
            elif key == 'volume_sma_periods': unique_params_values[key] = [20]
            elif key == 'rsi_periods': unique_params_values[key] = [14]

    indicator_map = {} # (지표종류, 파라미터값) -> 마스터 배열 인덱스
    current_idx = 0

    for length in unique_params_values['ema_lengths']:
        indicator_map[('ema', length)] = current_idx; current_idx += 1
    for period in unique_params_values['atr_periods']:
        indicator_map[('atr', period)] = current_idx; current_idx += 1
    for period in unique_params_values['adx_periods']:
        indicator_map[('adx', period)] = current_idx      # ADX
        indicator_map[('dmp', period)] = current_idx + 1  # +DI
        indicator_map[('dmn', period)] = current_idx + 2  # -DI
        current_idx += 3
    for period in unique_params_values['volume_sma_periods']:
        indicator_map[('vol_sma', period)] = current_idx; current_idx += 1
    for period in unique_params_values['rsi_periods']:
        indicator_map[('rsi', period)] = current_idx; current_idx += 1

    num_total_indicator_series = current_idx
    logger.info(f"고유 지표 파라미터 추출 및 매핑 완료. 총 {num_total_indicator_series}개의 고유 지표 시리즈 생성 예정.")
    logger.debug(f"Unique param values: {unique_params_values}")
    logger.debug(f"Indicator map: {indicator_map}")
    return unique_params_values, indicator_map, num_total_indicator_series


def precompute_all_indicators_for_gpu(df_ohlcv, unique_params_dict, interval_suffix, data_length):
    """주어진 OHLCV 데이터와 고유 파라미터 설정을 사용하여 모든 필요한 지표를 사전 계산하고, GPU 전송용 Numpy 배열로 반환합니다."""
    if df_ohlcv is None or df_ohlcv.empty:
        logger.warning(f"사전 계산을 위한 {interval_suffix} 데이터 없음."); return None

    _, temp_indicator_map, num_total_series = get_unique_indicator_params_and_map(param_ranges)

    master_indicator_array = np.full((num_total_series, data_length), np.nan, dtype=np.float64)
    logger.info(f"{interval_suffix} 데이터에 대한 마스터 지표 배열 생성 ({master_indicator_array.shape}).")

    for length in unique_params_dict.get('ema_lengths', []):
        if length > 0:
            idx = temp_indicator_map.get(('ema', length))
            if idx is not None:
                try:
                    ema_series = df_ohlcv.ta.ema(length=length, append=False)
                    if ema_series is not None and len(ema_series) == data_length: master_indicator_array[idx, :] = ema_series.to_numpy()
                except Exception as e: logger.error(f"EMA_{length}_{interval_suffix} 계산 오류: {e}")

    for period in unique_params_dict.get('atr_periods', []):
        if period > 0:
            idx = temp_indicator_map.get(('atr', period))
            if idx is not None:
                try:
                    atr_series = df_ohlcv.ta.atr(length=period, append=False)
                    if atr_series is not None and len(atr_series) == data_length: master_indicator_array[idx, :] = atr_series.to_numpy()
                except Exception as e: logger.error(f"ATR_{period}_{interval_suffix} 계산 오류: {e}")

    for period in unique_params_dict.get('adx_periods', []):
        if period > 0:
            adx_idx = temp_indicator_map.get(('adx', period))
            dmp_idx = temp_indicator_map.get(('dmp', period))
            dmn_idx = temp_indicator_map.get(('dmn', period))
            if adx_idx is not None and dmp_idx is not None and dmn_idx is not None:
                try:
                    adx_df = df_ohlcv.ta.adx(length=period, append=False)
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
                idx = temp_indicator_map.get(('vol_sma', period))
                if idx is not None:
                    try:
                        vol_sma_series = ta.sma(volume_series, length=period, append=False);
                        if vol_sma_series is not None and len(vol_sma_series) == data_length: master_indicator_array[idx, :] = vol_sma_series.to_numpy()
                    except Exception as e: logger.error(f"Volume SMA_{period}_{interval_suffix} 계산 오류: {e}")
    else:
        logger.warning(f"{interval_suffix} 데이터에 'Volume' 컬럼 없어 Volume SMA 계산 스킵.")

    for period in unique_params_dict.get('rsi_periods', []):
        if period > 0:
            idx = temp_indicator_map.get(('rsi', period))
            if idx is not None:
                try:
                    rsi_series = df_ohlcv.ta.rsi(length=period, append=False);
                    if rsi_series is not None and len(rsi_series) == data_length: master_indicator_array[idx, :] = rsi_series.to_numpy()
                except Exception as e: logger.error(f"RSI_{period}_{interval_suffix} 계산 오류: {e}")

    logger.info(f"{interval_suffix} 데이터에 대한 마스터 지표 배열 계산 완료.")
    return master_indicator_array


@cuda.jit
def run_batch_backtest_gpu_kernel(
    # --- Input Data Arrays (Device Arrays) ---
    open_prices_all, high_prices_all, low_prices_all, close_prices_all, volume_all,
    h1_indicators_all, h4_indicators_all, # [indicator_idx, time_idx]
    hour_of_day_all, day_of_week_all, # For temporal filter

    # --- Parameter Combinations (Device Arrays for the current BATCH) ---
    param_ema_short_h1_values, param_ema_long_h1_values, param_ema_htf_values,
    param_adx_period_values, param_adx_threshold_values,
    param_atr_period_sl_values, param_atr_multiplier_sl_values,
    param_risk_reward_ratio_values,
    param_use_htf_ema_filter_flags, param_use_adx_filter_flags, # 0 or 1
    param_risk_per_trade_percentage_values,
    param_exit_strategy_type_codes, # 0 for FixedRR, 1 for TrailingATR
    param_trailing_atr_period_values, param_trailing_atr_multiplier_values,
    param_use_volume_filter_flags, param_volume_sma_period_values, # 0 or 1
    param_use_rsi_filter_flags, param_rsi_period_values, # 0 or 1
    param_rsi_threshold_long_values, param_rsi_threshold_short_values,
    # Phase 1 Parameters
    param_adx_threshold_regime_values, param_atr_percent_threshold_regime_values,
    param_time_stop_period_hours_values, param_profit_threshold_for_trail_values,
    param_max_consecutive_losses_values, param_cooldown_period_bars_values,
    param_adx_threshold_for_short_values, param_price_breakdown_period_values,
    param_rsi_momentum_threshold_values,

    # --- Mapping from parameter value to index in h1_indicators_all/h4_indicators_all (Device Arrays for the current BATCH) ---
    h1_ema_short_indices, h1_ema_long_indices, h4_ema_htf_indices,
    h1_adx_indices, h1_atr_sl_indices, h1_atr_trail_indices,
    h1_vol_sma_indices, h1_rsi_indices,

    # --- Output Arrays (Device Arrays for the current BATCH) ---
    out_final_balances, out_total_trades, out_win_trades, out_error_flags, # 0 or 1
    out_pnl_percentages, out_profit_factors, out_max_drawdowns,

    # --- Scalar Config ---
    data_len: int,
    initial_balance_global: float,
    commission_global: float,
    slippage_global: float,
    min_trade_size_btc_global: float,
    quantity_precision_global: int,
    allowed_hours_bool_global, blocked_days_bool_global # For temporal filter
):
    combo_idx = cuda.grid(1)
    if combo_idx >= len(param_ema_short_h1_values): 
        return

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

    # Phase 1 Parameters
    adx_threshold_regime_p = param_adx_threshold_regime_values[combo_idx]
    atr_percent_threshold_regime_p = param_atr_percent_threshold_regime_values[combo_idx]
    time_stop_period_hours_p = param_time_stop_period_hours_values[combo_idx]
    profit_threshold_for_trail_p = param_profit_threshold_for_trail_values[combo_idx]
    max_consecutive_losses_p = param_max_consecutive_losses_values[combo_idx]
    cooldown_period_bars_p = param_cooldown_period_bars_values[combo_idx]
    adx_threshold_for_short_p = param_adx_threshold_for_short_values[combo_idx]
    price_breakdown_period_p = param_price_breakdown_period_values[combo_idx]
    rsi_momentum_threshold_p = param_rsi_momentum_threshold_values[combo_idx]

    idx_h1_ema_short = h1_ema_short_indices[combo_idx]
    idx_h1_ema_long = h1_ema_long_indices[combo_idx]
    idx_h4_ema_htf = h4_ema_htf_indices[combo_idx] 
    idx_h1_adx = h1_adx_indices[combo_idx]        
    idx_h1_atr_sl = h1_atr_sl_indices[combo_idx]
    idx_h1_atr_trail = h1_atr_trail_indices[combo_idx] 
    idx_h1_vol_sma = h1_vol_sma_indices[combo_idx]   
    idx_h1_rsi = h1_rsi_indices[combo_idx]        

    balance = initial_balance_global
    position = POSITION_NONE_GPU 
    entry_price_val = 0.0
    position_size_val = 0.0
    initial_stop_loss_val = 0.0 
    current_stop_loss_val = 0.0 
    take_profit_order_val = 0.0 
    entry_idx_val = -1 

    trade_count_local = 0
    win_count_local = 0
    error_flag_local = 0 

    peak_balance_local = initial_balance_global
    max_drawdown_local = 0.0
    gross_profit_local = 0.0
    gross_loss_local = 0.0

    # Circuit Breaker State (per combo_idx)
    consecutive_losses_local = 0
    is_in_cooldown_local = 0 # 0 for False, 1 for True
    cooldown_release_time_idx_local = -1 # Index of the candle when cooldown ends

    # Helper function for short entry conditions (confluence)
    def check_short_entry_conditions_gpu(current_idx, close_prices, low_prices, h1_indicators, 
                                        ema_short_idx, ema_long_idx, adx_idx, rsi_idx, 
                                        adx_threshold_for_short, price_breakdown_period, rsi_momentum_threshold):
        
        # Condition 1: Trend Structure (fast EMA below slow EMA)
        is_trend_bearish = h1_indicators[ema_short_idx, current_idx] < h1_indicators[ema_long_idx, current_idx]

        # Condition 2: Trend Strength (ADX > threshold for Short)
        is_trend_strong = h1_indicators[adx_idx, current_idx] > adx_threshold_for_short

        # Condition 3: Price Action Confirmation (recent low broken)
        # Need enough historical data for min() calculation
        if current_idx < price_breakdown_period:
            return False # Not enough data
        
        # Get the 'Low' prices for the last 'price_breakdown_period' candles excluding the current one
        min_val = low_prices[current_idx - price_breakdown_period] # Initialize with first element of slice
        for k in range(1, price_breakdown_period):
            if low_prices[current_idx - price_breakdown_period + k] < min_val:
                min_val = low_prices[current_idx - price_breakdown_period + k]
        
        min_low_in_period = min_val # This is the minimum of the slice
        is_price_breakdown = close_prices[current_idx] < min_low_in_period

        # Condition 4: Momentum Oscillator (RSI < threshold)
        is_momentum_bearish = h1_indicators[rsi_idx, current_idx] < rsi_momentum_threshold

        return is_trend_bearish and is_trend_strong and is_price_breakdown and is_momentum_bearish

    for i in range(1, data_len): 
        curr_open = open_prices_all[i]; curr_high = high_prices_all[i]; curr_low = low_prices_all[i]; curr_close = close_prices_all[i]; curr_volume = volume_all[i]

        curr_ema_short_h1 = h1_indicators_all[idx_h1_ema_short, i]
        prev_ema_short_h1 = h1_indicators_all[idx_h1_ema_short, i-1]
        curr_ema_long_h1 = h1_indicators_all[idx_h1_ema_long, i]
        prev_ema_long_h1 = h1_indicators_all[idx_h1_ema_long, i-1]
        curr_atr_sl = h1_indicators_all[idx_h1_atr_sl, i]

        curr_adx = math.nan
        if use_adx_filter_f and idx_h1_adx >= 0:
             curr_adx = h1_indicators_all[idx_h1_adx, i]

        curr_ema_htf = math.nan
        if use_htf_ema_filter_f and idx_h4_ema_htf >= 0: 
            curr_ema_htf = h4_indicators_all[idx_h4_ema_htf, i]

        curr_atr_trail = math.nan
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and idx_h1_atr_trail >=0: 
            curr_atr_trail = h1_indicators_all[idx_h1_atr_trail, i]

        curr_vol_sma = math.nan
        if use_volume_filter_f and idx_h1_vol_sma >=0: 
            curr_vol_sma = h1_indicators_all[idx_h1_vol_sma, i]

        curr_rsi = math.nan
        if use_rsi_filter_f and idx_h1_rsi >=0: 
            curr_rsi = h1_indicators_all[idx_h1_rsi, i]

        if math.isnan(curr_ema_short_h1) or math.isnan(curr_ema_long_h1) or \
           math.isnan(prev_ema_short_h1) or math.isnan(prev_ema_long_h1) or \
           math.isnan(curr_atr_sl) :
            continue
        if use_adx_filter_f and math.isnan(curr_adx): continue
        if use_htf_ema_filter_f and math.isnan(curr_ema_htf): continue
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and math.isnan(curr_atr_trail): continue
        if use_volume_filter_f and math.isnan(curr_vol_sma): continue
        if use_rsi_filter_f and math.isnan(curr_rsi): continue

        if curr_atr_sl <= 0: continue 
        if exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and curr_atr_trail <= 0: continue

        if position != POSITION_NONE_GPU:
            exit_reason_code = EXIT_REASON_NONE_GPU 
            exit_price = 0.0

            # Calculate current profit in USD
            current_profit_usd = 0.0
            if position == POSITION_LONG_GPU:
                current_profit_usd = (curr_close - entry_price_val) * position_size_val
            elif position == POSITION_SHORT_GPU:
                current_profit_usd = (entry_price_val - curr_close) * position_size_val

            # 1. Time Stop
            position_held_bars = i - entry_idx_val
            if position_held_bars >= time_stop_period_hours_p and current_profit_usd < 0:
                exit_price = curr_close # Exit at current close
                exit_reason_code = EXIT_REASON_TIME_STOP_GPU
            
            # 2. ATR Trailing Stop (activate only when profitable enough)
            if exit_reason_code == EXIT_REASON_NONE_GPU and exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU and current_profit_usd >= profit_threshold_for_trail_p:
                if position == POSITION_LONG_GPU:
                    new_trail_sl = curr_high - (curr_atr_trail * trailing_atr_multiplier_p) 
                    current_stop_loss_val = max(current_stop_loss_val, new_trail_sl)
                elif position == POSITION_SHORT_GPU:
                    new_trail_sl = curr_low + (curr_atr_trail * trailing_atr_multiplier_p) 
                    current_stop_loss_val = min(current_stop_loss_val, new_trail_sl)

            # Check for SL/TP hit (including time stop and trailing stop)
            if exit_reason_code == EXIT_REASON_NONE_GPU: # Only check if not already exited by time stop
                if position == POSITION_LONG_GPU:
                    if curr_low <= current_stop_loss_val: 
                        exit_price = current_stop_loss_val
                        exit_reason_code = EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                    elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_high >= take_profit_order_val: 
                        exit_price = take_profit_order_val
                        exit_reason_code = EXIT_REASON_TP_GPU
                elif position == POSITION_SHORT_GPU:
                    if curr_high >= current_stop_loss_val: 
                        exit_price = current_stop_loss_val
                        exit_reason_code = EXIT_REASON_SL_GPU if current_stop_loss_val == initial_stop_loss_val else EXIT_REASON_TRAIL_SL_GPU
                    elif exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU and curr_low <= take_profit_order_val: 
                        exit_price = take_profit_order_val
                        exit_reason_code = EXIT_REASON_TP_GPU

            if exit_reason_code != EXIT_REASON_NONE_GPU:
                gross_pnl = 0.0
                if position == POSITION_LONG_GPU: gross_pnl = (exit_price - entry_price_val) * position_size_val
                else: gross_pnl = (entry_price_val - exit_price) * position_size_val

                entry_value = entry_price_val * position_size_val
                exit_value = exit_price * position_size_val
                entry_cost = entry_value * (commission_global + slippage_global) 
                exit_cost = exit_value * (commission_global + slippage_global)  
                total_trade_cost = entry_cost + exit_cost
                net_pnl = gross_pnl - total_trade_cost

                balance += net_pnl

                if balance > peak_balance_local: peak_balance_local = balance
                drawdown = (peak_balance_local - balance) / peak_balance_local if peak_balance_local > 0 else 0.0
                if drawdown > max_drawdown_local: max_drawdown_local = drawdown

                if gross_pnl > 0: gross_profit_local += gross_pnl 
                else: gross_loss_local += abs(gross_pnl)

                trade_count_local += 1
                if net_pnl > 0: win_count_local += 1

                # Circuit Breaker: Update consecutive losses
                if net_pnl <= 0: # Loss or zero PnL
                    consecutive_losses_local += 1
                else:
                    consecutive_losses_local = 0 # Reset on win
                
                if consecutive_losses_local >= max_consecutive_losses_p: # Check if cooldown should start
                    is_in_cooldown_local = 1
                    cooldown_release_time_idx_local = i + cooldown_period_bars_p # Cooldown ends after X bars

                if balance <= 0: error_flag_local = 1; break 

                position = POSITION_NONE_GPU; entry_price_val = 0.0; position_size_val = 0.0
                initial_stop_loss_val = 0.0; current_stop_loss_val = 0.0; take_profit_order_val = 0.0
                entry_idx_val = -1

        if position == POSITION_NONE_GPU and error_flag_local == 0:
            # --- Phase 1 Filters (before signal generation) ---
            # if is_in_cooldown_local == 1: # If currently in cooldown
            #     if i >= cooldown_release_time_idx_local: # If cooldown period has passed
            #         is_in_cooldown_local = 0 # Exit cooldown
            #         consecutive_losses_local = 0 # Reset losses
            #     else:
            #         continue # Still in cooldown, skip signal generation

            # # 2. Temporal Filter Check
            # curr_hour = hour_of_day_all[i]
            # curr_weekday = day_of_week_all[i]
            # if allowed_hours_bool_global[curr_hour] == 0: # If current hour is not allowed
            #     continue
            # if blocked_days_bool_global[curr_weekday] == 1: # If current day is blocked
            #     continue

            # # 3. Regime Filter Check
            # # Need ADX and ATR values from h1_indicators_all
            # curr_adx_regime = h1_indicators_all[idx_h1_adx, i] if use_adx_filter_f and idx_h1_adx >= 0 else math.nan
            # curr_atr_regime = h1_indicators_all[idx_h1_atr_sl, i] # Using atr_sl for regime filter ATR
            
            # if math.isnan(curr_adx_regime) or math.isnan(curr_atr_regime) or curr_close <= 0:
            #     # Not enough data for regime filter, skip
            #     continue

            # atr_percent_value = (curr_atr_regime / curr_close) * 100

            # if not (curr_adx_regime >= adx_threshold_regime_p and atr_percent_value >= atr_percent_threshold_regime_p):
            #     # Not a favorable regime, skip
            #     continue
            # --- End Phase 1 Filters ---

            long_signal = False
            short_signal = False

            if prev_ema_short_h1 < prev_ema_long_h1 and curr_ema_short_h1 > curr_ema_long_h1:
                long_signal = True
            elif prev_ema_short_h1 > prev_ema_long_h1 and curr_ema_short_h1 < curr_ema_long_h1:
                # Apply confluence filter for short entry
                if check_short_entry_conditions_gpu(i, close_prices_all, low_prices_all, h1_indicators_all,
                                                    idx_h1_ema_short, idx_h1_ema_long, idx_h1_adx, idx_h1_rsi,
                                                    adx_threshold_for_short_p, price_breakdown_period_p, rsi_momentum_threshold_p):
                    short_signal = True

            if long_signal or short_signal:
                if use_htf_ema_filter_f:
                    if long_signal and curr_close < curr_ema_htf: long_signal = False
                    if short_signal and curr_close > curr_ema_htf: short_signal = False

                if use_adx_filter_f:
                    if curr_adx < adx_threshold_p : long_signal, short_signal = False, False

                if use_volume_filter_f:
                    if curr_volume <= curr_vol_sma : long_signal, short_signal = False, False 

                if use_rsi_filter_f:
                    if long_signal and curr_rsi < rsi_threshold_long_p: long_signal = False
                    if short_signal and curr_rsi > rsi_threshold_short_p: short_signal = False 

            signal_type = POSITION_NONE_GPU
            if long_signal: signal_type = POSITION_LONG_GPU
            elif short_signal: signal_type = POSITION_SHORT_GPU

            if signal_type != POSITION_NONE_GPU:
                entry_p = curr_close 
                sl_distance = curr_atr_sl * atr_multiplier_sl_p
                if sl_distance <= 0: continue 

                potential_initial_sl = 0.0
                potential_tp = 0.0

                if signal_type == POSITION_LONG_GPU:
                    potential_initial_sl = entry_p - sl_distance
                    if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU:
                        potential_tp = entry_p + (sl_distance * risk_reward_ratio_p)
                else: 
                    potential_initial_sl = entry_p + sl_distance
                    if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU:
                        potential_tp = entry_p - (sl_distance * risk_reward_ratio_p)

                valid_entry_conditions = False
                if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU:
                    if signal_type == POSITION_LONG_GPU and potential_tp > entry_p and potential_initial_sl < entry_p:
                        valid_entry_conditions = True
                    elif signal_type == POSITION_SHORT_GPU and potential_tp < entry_p and potential_initial_sl > entry_p:
                        valid_entry_conditions = True
                elif exit_strategy_type_c == EXIT_STRATEGY_TRAILING_ATR_GPU: 
                     if signal_type == POSITION_LONG_GPU and potential_initial_sl < entry_p : valid_entry_conditions = True
                     elif signal_type == POSITION_SHORT_GPU and potential_initial_sl > entry_p : valid_entry_conditions = True


                if valid_entry_conditions:
                    risk_amount_per_trade = balance * risk_per_trade_percentage_p
                    position_size_calc = risk_amount_per_trade / sl_distance 

                    power_factor = 1
                    for _ in range(quantity_precision_global): power_factor *= 10
                    calculated_size = math.floor(position_size_calc * power_factor) / power_factor

                    if calculated_size >= min_trade_size_btc_global:
                        position = signal_type
                        entry_price_val = entry_p
                        position_size_val = calculated_size
                        initial_stop_loss_val = potential_initial_sl
                        current_stop_loss_val = potential_initial_sl 
                        if exit_strategy_type_c == EXIT_STRATEGY_FIXED_RR_GPU:
                            take_profit_order_val = potential_tp
                        else:
                            take_profit_order_val = 0.0 
                        entry_idx_val = i

    out_final_balances[combo_idx] = balance
    out_total_trades[combo_idx] = trade_count_local
    out_win_trades[combo_idx] = win_count_local
    out_error_flags[combo_idx] = error_flag_local

    total_pnl = balance - initial_balance_global
    out_pnl_percentages[combo_idx] = (total_pnl / initial_balance_global) * 100 if initial_balance_global > 0 else 0.0
    out_profit_factors[combo_idx] = gross_profit_local / gross_loss_local if gross_loss_local > 0 else (float('inf') if gross_profit_local > 0 else 0.0)
    out_max_drawdowns[combo_idx] = max_drawdown_local * 100 


def print_performance_report(performance, params_id="N/A"):
    """백테스팅 성과를 로거를 통해 출력합니다."""
    logger.info(f"\n--- 백테스팅 성과 보고서 (ID: {params_id}) ---")
    if not performance:
        logger.info("성과 데이터 없음.")
        return

    if "error" in performance and performance["error"]:
        logger.error(f"백테스팅 오류 발생! (ID: {params_id})")
        return

    report_map = {
        "initial_balance": "Initial Balance",
        "final_balance": "Final Balance",
        "total_net_pnl": "Total Net Pnl",
        "total_net_pnl_percentage": "Total Net Pnl Percentage",
        "num_trades": "Num Trades",
        "num_wins": "Num Wins",
        "num_losses": "Num Losses",
        "win_rate_percentage": "Win Rate Percentage",
        "profit_factor": "Profit Factor",
        "max_drawdown_percentage": "Max Drawdown Percentage"
    }
    report_lines = [f"{display_name:<28}: {performance.get(key, 'N/A')}"
                    for key, display_name in report_map.items()]

    logger.info("\n".join(report_lines))
    logger.info(f"--- 보고서 종료 (ID: {params_id}) ---\n")


# --- Main Execution Logic ---
if __name__ == "__main__":
    logger.info("=== EMA Crossover 전략 백테스터 V5.11 (최적화 파라미터 범위) 실행 시작 ===")
    overall_start_time = time.time()

    # --- 헤드투헤드 실행 오케스트레이터: RUN_BOTH=1이면 baseline/improved 두 번 자동 실행 후 종료 ---
    try:
        if os.getenv('RUN_BOTH', '0') == '1':
            import sys, subprocess
            preset = os.getenv('PERIOD_PRESET', 'full')
            logger.info(f"헤드투헤드 실행 시작 (PERIOD_PRESET={preset}) → baseline -> improved 순서")
            for prof in ('baseline', 'improved'):
                env = os.environ.copy()
                env['PROFILE_NAME'] = prof
                env['PERIOD_PRESET'] = preset
                env['JSON_TAG'] = f"{prof}_{preset}"
                logger.info(f"서브프로세스 실행: PROFILE_NAME={prof}, PERIOD_PRESET={preset}")
                try:
                    subprocess.run([sys.executable, __file__], env=env, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"서브프로세스 실패(PROFILE={prof}): {e}")
            logger.info("헤드투헤드 실행 완료. 부모 프로세스 종료.")
            exit()
    except Exception as e:
        logger.error(f"헤드투헤드 오케스트레이션 오류: {e}")

    try:
        if OFFLINE_MODE:
            # 오프라인이라도 GPU 가용시 커널 실행은 가능
            if not cuda.is_available() or len(cuda.gpus) == 0:
                logger.warning("CUDA 사용 불가 또는 GPU 없음. 대량 조합에서는 실행 시간이 길 수 있습니다.")
            else:
                logger.info(f"CUDA 사용 가능. GPU 장치 수: {len(cuda.gpus)}")
                selected_gpu = cuda.get_current_device(); logger.info(f"선택된 GPU: {selected_gpu.name.decode()}")
                cp.cuda.runtime.getDeviceCount(); logger.info("CuPy도 CUDA 장치 인식 완료.")
        else:
            if not cuda.is_available(): logger.error("CUDA 사용 불가능."); exit()
            if len(cuda.gpus) == 0: logger.error("사용 가능한 GPU 장치 없음."); exit()
            logger.info(f"CUDA 사용 가능. GPU 장치 수: {len(cuda.gpus)}")
            selected_gpu = cuda.get_current_device(); logger.info(f"선택된 GPU: {selected_gpu.name.decode()}")
            cp.cuda.runtime.getDeviceCount(); logger.info("CuPy도 CUDA 장치 인식 완료.")
    except Exception as e: logger.error(f"CUDA/CuPy 초기화 오류: {e}."); exit()

    logger.info(f"주 시간봉 ({interval_primary_bt}) 데이터 로딩 중...")
    if OFFLINE_MODE and OFFLINE_OHLCV_H1:
        hist_df_primary_master_ohlcv = _read_ohlcv_local(OFFLINE_OHLCV_H1)
    else:
        hist_df_primary_master_ohlcv = get_historical_data(symbol_backtest, interval_primary_bt, start_str=start_date_str, end_str=end_date_str, max_klines=num_klines_to_fetch)
    if hist_df_primary_master_ohlcv is None or hist_df_primary_master_ohlcv.empty: logger.error("주 시간봉 데이터 로딩 실패."); exit()
    data_length_global = len(hist_df_primary_master_ohlcv)
    df_timestamps_for_results = hist_df_primary_master_ohlcv.index 

    needs_htf_data_globally = 'use_htf_ema_filter' in param_ranges and True in param_ranges['use_htf_ema_filter']
    hist_df_htf_master_ohlcv = None
    if needs_htf_data_globally:
        first_primary_timestamp = hist_df_primary_master_ohlcv.index.min()
        max_lookback_needed_for_htf = 0
        if param_ranges.get('ema_htf'): 
            max_lookback_needed_for_htf = max(param_ranges.get('ema_htf', [0])) 

        htf_additional_klines = max(max_lookback_needed_for_htf * 3, 200) 

        try:
            interval_str_for_pd = interval_htf_bt.replace('m', 'T')
            if 'h' in interval_htf_bt: interval_str_for_pd = interval_htf_bt.upper()
            elif 'd' in interval_htf_bt: interval_str_for_pd = interval_htf_bt.upper()
            elif 'w' in interval_htf_bt: interval_str_for_pd = interval_htf_bt.upper()
            htf_interval_delta = pd.Timedelta(interval_str_for_pd)
        except ValueError:
            logger.warning(f"HTF interval '{interval_htf_bt}' 파싱 실패. 기본값 4시간 사용.")
            htf_interval_delta = timedelta(hours=4) 

        htf_fetch_start_dt = first_primary_timestamp - (htf_interval_delta * htf_additional_klines)
        htf_fetch_start_str = htf_fetch_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        htf_end_str_calc = hist_df_primary_master_ohlcv.index.max().strftime("%Y-%m-%d %H:%M:%S") 

        logger.info(f"상위 시간봉 ({interval_htf_bt}) 데이터 로딩 중 ({htf_fetch_start_str} ~ {htf_end_str_calc})...")
        if OFFLINE_MODE and OFFLINE_OHLCV_H4:
            hist_df_htf_master_ohlcv_raw = _read_ohlcv_local(OFFLINE_OHLCV_H4)
        else:
            hist_df_htf_master_ohlcv_raw = get_historical_data(symbol_backtest, interval_htf_bt, start_str=htf_fetch_start_str, end_str=htf_end_str_calc)

        if hist_df_htf_master_ohlcv_raw is not None and not hist_df_htf_master_ohlcv_raw.empty:
            hist_df_htf_master_ohlcv_raw.sort_index(inplace=True)
            temp_df_primary_time = pd.DataFrame(index=df_timestamps_for_results) 

            if temp_df_primary_time.index.tz != hist_df_htf_master_ohlcv_raw.index.tz:
                 if temp_df_primary_time.index.tz is None: temp_df_primary_time.index = temp_df_primary_time.index.tz_localize('UTC') 
                 if hist_df_htf_master_ohlcv_raw.index.tz is None: hist_df_htf_master_ohlcv_raw.index = hist_df_htf_master_ohlcv_raw.index.tz_localize('UTC')

            hist_df_htf_master_ohlcv = pd.merge_asof(
                temp_df_primary_time.sort_index(),
                hist_df_htf_master_ohlcv_raw.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward' 
            )
            if len(hist_df_htf_master_ohlcv) != data_length_global:
                logger.warning(f"HTF/Primary 데이터 길이 불일치 발생 후 재정렬 시도. Primary: {data_length_global}, HTF 정렬 후: {len(hist_df_htf_master_ohlcv)}")
                hist_df_htf_master_ohlcv = hist_df_htf_master_ohlcv.reindex(df_timestamps_for_results, method='ffill')
                if len(hist_df_htf_master_ohlcv) != data_length_global:
                     logger.error("HTF 데이터 길이 최종 불일치. HTF 필터 사용 불가할 수 있음.")
            del hist_df_htf_master_ohlcv_raw 
        else:
            logger.warning(f"상위 시간봉({interval_htf_bt}) 데이터 로드 실패.")
    else:
        logger.info("HTF EMA 필터 사용 조합 없어 HTF 데이터 로딩 스킵.")


    unique_indicator_configs, indicator_to_idx_map, num_total_indicator_series_global = get_unique_indicator_params_and_map(param_ranges)
    logger.info("마스터 H1 지표 사전 계산 (GPU용 배열)...")
    master_h1_indicators_np = precompute_all_indicators_for_gpu(hist_df_primary_master_ohlcv, unique_indicator_configs, "H1", data_length_global)
    if master_h1_indicators_np is None: logger.error("H1 마스터 지표 생성 실패."); exit()

    master_h4_indicators_np = None
    if hist_df_htf_master_ohlcv is not None and not hist_df_htf_master_ohlcv.empty:
        logger.info("마스터 H4 지표 사전 계산 (GPU용 배열)...")
        master_h4_indicators_np = precompute_all_indicators_for_gpu(hist_df_htf_master_ohlcv, unique_indicator_configs, "H4", data_length_global)
        if master_h4_indicators_np is None: logger.warning("H4 마스터 지표 생성 실패.")
    elif needs_htf_data_globally: 
        logger.warning("HTF 데이터가 필요했으나 준비되지 않아 H4 지표 배열은 NaN으로 채워진 H1 배열 크기를 사용합니다.")
        master_h4_indicators_np = np.full_like(master_h1_indicators_np, np.nan) if master_h1_indicators_np is not None else None
    else: 
        logger.info("HTF 데이터 불필요. H4 지표 배열은 H1 지표 배열의 NaN 복사본 사용 (커널에서 사용 안 함).")
        master_h4_indicators_np = np.full_like(master_h1_indicators_np, np.nan) if master_h1_indicators_np is not None else None


    open_p_np = hist_df_primary_master_ohlcv['Open'].to_numpy(dtype=np.float64)
    high_p_np = hist_df_primary_master_ohlcv['High'].to_numpy(dtype=np.float64)
    low_p_np = hist_df_primary_master_ohlcv['Low'].to_numpy(dtype=np.float64)
    close_p_np = hist_df_primary_master_ohlcv['Close'].to_numpy(dtype=np.float64)
    volume_np = hist_df_primary_master_ohlcv['Volume'].to_numpy(dtype=np.float64)

    # Temporal Filter Data Preparation
    hour_of_day_np = np.array([ts.hour for ts in df_timestamps_for_results], dtype=np.int8)
    day_of_week_np = np.array([ts.weekday() for ts in df_timestamps_for_results], dtype=np.int8) # Monday=0, Sunday=6

    # Fixed values for temporal filter (from real_M1.py defaults)
    # Convert sets to boolean arrays for GPU
    allowed_hours_fixed = {10, 12, 14, 17}
    blocked_days_fixed = {'Monday'}

    allowed_hours_bool_np = np.zeros(24, dtype=np.int8) # 0-23 hours
    for h in allowed_hours_fixed:
        allowed_hours_bool_np[h] = 1

    blocked_days_bool_np = np.zeros(7, dtype=np.int8) # 0-6 for Mon-Sun
    day_name_to_weekday = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    for d_name in blocked_days_fixed:
        if d_name in day_name_to_weekday:
            blocked_days_bool_np[day_name_to_weekday[d_name]] = 1

    del hist_df_primary_master_ohlcv, hist_df_htf_master_ohlcv; gc.collect()

# --- 의존성-인식 파라미터 조합 생성기 ---
def iter_dependent_param_combos(pr):
    import itertools as it

    # 공통 축 (항상 포함)
    base_keys = [
        'ema_short_h1','ema_long_h1','atr_period_sl','atr_multiplier_sl',
        'risk_per_trade_percentage',
        # Regime / Circuit / Phase-1 (항상 활성)
        'adx_threshold_regime','atr_percent_threshold_regime',
        'time_stop_period_hours','profit_threshold_for_trail',
        'max_consecutive_losses','cooldown_period_bars',
        # Short 전용
        'adx_threshold_for_short','price_breakdown_period','rsi_momentum_threshold',
    ]
    base_vals = [pr[k] for k in base_keys]

    for vals in it.product(*base_vals):
        base = dict(zip(base_keys, vals))

        # EMA 제약: short < long
        if base['ema_short_h1'] >= base['ema_long_h1']:
            continue

        # 조건부 축
        # ADX
        adx_choices = [{'use_adx_filter': False, 'adx_period': pr['adx_period'][0], 'adx_threshold': pr['adx_threshold'][0]}]
        if True in pr.get('use_adx_filter',[False]):
            adx_choices = [{'use_adx_filter': True, 'adx_period': p, 'adx_threshold': t}
                           for p,t in it.product(pr['adx_period'], pr['adx_threshold'])]

        # Volume
        vol_choices = [{'use_volume_filter': False, 'volume_sma_period': pr['volume_sma_period'][0]}]
        if True in pr.get('use_volume_filter',[False]):
            vol_choices = [{'use_volume_filter': True, 'volume_sma_period': p}
                           for p in pr['volume_sma_period']]

        # RSI
        rsi_choices = [{'use_rsi_filter': False, 'rsi_period': pr['rsi_period'][0],
                        'rsi_threshold_long': pr['rsi_threshold_long'][0],
                        'rsi_threshold_short': pr['rsi_threshold_short'][0]}]
        if True in pr.get('use_rsi_filter',[False]):
            rsi_choices = [{'use_rsi_filter': True, 'rsi_period': p,
                            'rsi_threshold_long': tl, 'rsi_threshold_short': ts}
                           for p,tl,ts in it.product(pr['rsi_period'], pr['rsi_threshold_long'], pr['rsi_threshold_short'])]

        # HTF EMA
        htf_choices = [{'use_htf_ema_filter': False, 'ema_htf': pr['ema_htf'][0]}]
        if True in pr.get('use_htf_ema_filter',[False]):
            htf_choices = [{'use_htf_ema_filter': True, 'ema_htf': p} for p in pr['ema_htf']]

        # 조건부 묶음 합치기
        for adx_c, vol_c, rsi_c, htf_c in it.product(adx_choices, vol_choices, rsi_choices, htf_choices):
            mid = {**base, **adx_c, **vol_c, **rsi_c, **htf_c}

            # --- Exit 전략: "분기 합"
            # FixedRR 브랜치
            for rr in pr['risk_reward_ratio']:
                out = mid.copy()
                out['exit_strategy_type'] = 'FixedRR'
                out['risk_reward_ratio'] = rr
                # 사용 안하는 트레일링 값은 ID 안정화용 기본값
                out['trailing_atr_period'] = pr['trailing_atr_period'][0]
                out['trailing_atr_multiplier'] = pr['trailing_atr_multiplier'][0]
                yield out

            # TrailingATR 브랜치
            for t_per, t_mul in it.product(pr['trailing_atr_period'], pr['trailing_atr_multiplier']):
                out = mid.copy()
                out['exit_strategy_type'] = 'TrailingATR'
                out['trailing_atr_period'] = t_per
                out['trailing_atr_multiplier'] = t_mul
                # 사용 안하는 RR은 ID 안정화용 기본값
                out['risk_reward_ratio'] = pr['risk_reward_ratio'][0]
                yield out

    logger.info("파라미터 조합 생성 및 GPU 배열 준비 시작 (점진적 처리)...")
    prep_start_time = time.time()

    param_keys, param_values_iter_list = zip(*param_ranges.items())


    # --- 파라미터 조합 생성 및 GPU 배열 준비 (스트리밍 처리) ---
    logger.info("파라미터 조합 생성 및 GPU 배열 준비 시작 (스트리밍 처리).")
    prep_start_time = time.time()

    all_results_summary = [] # 모든 배치 결과를 수집할 리스트
    total_combinations_processed = 0
    
    # 제너레이터에서 직접 조합을 가져옵니다.
    param_generator = iter_dependent_param_combos(param_ranges)

    # 배치 처리를 위한 루프
    batch_num = 0
    while True:
        batch_num += 1
        current_batch_param_dicts = []
        current_batch_param_ids = []
        
        # 배치에 필요한 파라미터 값들을 저장할 임시 리스트
        batch_p_ema_s_vals, batch_p_ema_l_vals, batch_p_ema_h_vals = [], [], []
        batch_p_adx_p_vals, batch_p_adx_t_vals = [], []
        batch_p_atr_p_sl_vals, batch_p_atr_m_sl_vals = [], []
        batch_p_rr_vals, batch_p_use_htf_f_vals, batch_p_use_adx_f_vals = [], [], []
        batch_p_risk_vals, batch_p_exit_type_vals = [], []
        batch_p_trail_atr_p_vals, batch_p_trail_atr_m_vals = [], []
        batch_p_use_vol_f_vals, batch_p_vol_sma_p_vals = [], []
        batch_p_use_rsi_f_vals, batch_p_rsi_p_vals, batch_p_rsi_l_vals, batch_p_rsi_s_vals = [], [], [], []
        batch_p_use_regime_f_vals = []
        batch_p_adx_t_regime_vals, batch_p_atr_p_regime_vals = [], []
        batch_p_time_stop_h_vals, batch_p_profit_trail_vals = [], []
        batch_p_max_consec_l_vals, batch_p_cooldown_b_vals = [], []
        batch_p_adx_t_short_vals, batch_p_price_bd_p_vals, batch_p_rsi_mom_t_vals = [], [], []
        
        batch_idx_h1_ema_s, batch_idx_h1_ema_l, batch_idx_h4_ema_h = [], [], []
        batch_idx_h1_adx, batch_idx_h1_atr_sl, batch_idx_h1_atr_trail = [], [], []
        batch_idx_h1_vol_sma, batch_idx_h1_rsi = [], []

        # 현재 배치를 채웁니다.
        batch_fill_start_time = time.time()
        for _ in range(BATCH_SIZE):
            try:
                combo = next(param_generator)
                total_combinations_processed += 1

                # param_id 생성 (기존 로직과 동일)
                param_id_parts = []
                c_id = combo
                param_id_parts.append(f"EMA{c_id['ema_short_h1']}-{c_id['ema_long_h1']}")
                if c_id.get('use_adx_filter', False): param_id_parts.append(f"ADX{c_id['adx_period']}_{c_id['adx_threshold']}")
                if c_id.get('use_htf_ema_filter', False): param_id_parts.append(f"H4Filt{c_id['ema_htf']}")
                param_id_parts.append(f"SL{c_id['atr_multiplier_sl']}ATR{c_id['atr_period_sl']}")
                if c_id['exit_strategy_type'] == 'FixedRR': param_id_parts.append(f"RR{c_id['risk_reward_ratio']}")
                else: param_id_parts.append(f"Trail{c_id['trailing_atr_multiplier']}ATR{c_id['trailing_atr_period']}")
                if c_id.get('use_volume_filter', False): param_id_parts.append(f"VolFilt{c_id['volume_sma_period']}")
                if c_id.get('use_rsi_filter', False): param_id_parts.append(f"RSI{c_id['rsi_period']}_{c_id['rsi_threshold_long']}-{c_id['rsi_threshold_short']}")
                param_id_parts.append(f"Risk{c_id['risk_per_trade_percentage']*100:.1f}%")
                param_id_str = "_".join(map(str, param_id_parts))

                current_batch_param_dicts.append(combo)
                current_batch_param_ids.append(param_id_str)

                # 배치별 파라미터 값 리스트에 추가
                batch_p_ema_s_vals.append(combo['ema_short_h1']); batch_p_ema_l_vals.append(combo['ema_long_h1']); batch_p_ema_h_vals.append(combo['ema_htf'])
                batch_p_adx_p_vals.append(combo['adx_period']); batch_p_adx_t_vals.append(combo['adx_threshold']); batch_p_atr_p_sl_vals.append(combo['atr_period_sl']); batch_p_atr_m_sl_vals.append(combo['atr_multiplier_sl'])
                batch_p_rr_vals.append(combo['risk_reward_ratio']); batch_p_use_htf_f_vals.append(1 if combo.get('use_htf_ema_filter') else 0); batch_p_use_adx_f_vals.append(1 if combo.get('use_adx_filter') else 0)
                batch_p_risk_vals.append(combo['risk_per_trade_percentage']); batch_p_exit_type_vals.append(EXIT_STRATEGY_FIXED_RR_GPU if combo['exit_strategy_type'] == 'FixedRR' else EXIT_STRATEGY_TRAILING_ATR_GPU)
                batch_p_trail_atr_p_vals.append(combo['trailing_atr_period']); batch_p_trail_atr_m_vals.append(combo['trailing_atr_multiplier'])
                batch_p_use_vol_f_vals.append(1 if combo.get('use_volume_filter') else 0); batch_p_vol_sma_p_vals.append(combo['volume_sma_period'])
                batch_p_use_rsi_f_vals.append(1 if combo.get('use_rsi_filter') else 0); batch_p_rsi_p_vals.append(combo['rsi_period']); batch_p_rsi_l_vals.append(combo['rsi_threshold_long']); batch_p_rsi_s_vals.append(combo['rsi_threshold_short'])
                batch_p_use_regime_f_vals.append(1 if combo.get('use_regime_filter') else 0)
                batch_p_adx_t_regime_vals.append(combo['adx_threshold_regime']); batch_p_atr_p_regime_vals.append(combo['atr_percent_threshold_regime'])
                batch_p_time_stop_h_vals.append(combo['time_stop_period_hours']); batch_p_profit_trail_vals.append(combo['profit_threshold_for_trail'])
                batch_p_max_consec_l_vals.append(combo['max_consecutive_losses']); batch_p_cooldown_b_vals.append(combo['cooldown_period_bars'])
                batch_p_adx_t_short_vals.append(combo['adx_threshold_for_short']); batch_p_price_bd_p_vals.append(combo['price_breakdown_period']); batch_p_rsi_mom_t_vals.append(combo['rsi_momentum_threshold'])
                
                batch_idx_h1_ema_s.append(indicator_to_idx_map.get(('ema', combo['ema_short_h1']), -1))
                batch_idx_h1_ema_l.append(indicator_to_idx_map.get(('ema', combo['ema_long_h1']), -1))
                batch_idx_h4_ema_h.append(indicator_to_idx_map.get(('ema', combo['ema_htf']), -1) if combo.get('use_htf_ema_filter') else -1)
                batch_idx_h1_adx.append(indicator_to_idx_map.get(('adx', combo['adx_period']), -1) if combo.get('use_adx_filter') else -1)
                batch_idx_h1_atr_sl.append(indicator_to_idx_map.get(('atr', combo['atr_period_sl']), -1))
                batch_idx_h1_atr_trail.append(indicator_to_idx_map.get(('atr', combo['trailing_atr_period']), -1) if combo['exit_strategy_type'] == 'TrailingATR' else -1)
                batch_idx_h1_vol_sma.append(indicator_to_idx_map.get(('vol_sma', combo['volume_sma_period']), -1) if combo.get('use_volume_filter') else -1)
                batch_idx_h1_rsi.append(indicator_map.get(('rsi', combo['rsi_period']), -1) if combo.get('use_rsi_filter') else -1)

            except StopIteration:
                # 제너레이터 소진: 마지막 배치 처리 후 루프 종료
                break
        
        current_batch_size = len(current_batch_param_dicts)
        if current_batch_size == 0:
            break # 현재 배치가 비어있으면 루프 종료

        logger.info(f"--- 배치 {batch_num} (크기: {current_batch_size}) 처리 시작 ---")
        
        # 배치별 NumPy 배열 생성 및 GPU 전송
        batch_prep_start = time.time()
        np_p_ema_s_vals_batch=np.array(batch_p_ema_s_vals,dtype=np.int32); np_p_ema_l_vals_batch=np.array(batch_p_ema_l_vals,dtype=np.int32); np_p_ema_h_vals_batch=np.array(batch_p_ema_h_vals,dtype=np.int32)
        np_p_adx_p_vals_batch=np.array(batch_p_adx_p_vals,dtype=np.int32); np_p_adx_t_vals_batch=np.array(batch_p_adx_t_vals,dtype=np.float64); np_p_atr_p_sl_vals_batch=np.array(batch_p_atr_p_sl_vals,dtype=np.int32)
        np_p_atr_m_sl_vals_batch=np.array(batch_p_atr_m_sl_vals,dtype=np.float64); np_p_rr_vals_batch=np.array(batch_p_rr_vals,dtype=np.float64); np_p_use_htf_f_vals_batch=np.array(batch_p_use_htf_f_vals,dtype=np.int8)
        np_p_use_adx_f_vals_batch=np.array(batch_p_use_adx_f_vals,dtype=np.int8); np_p_risk_vals_batch=np.array(batch_p_risk_vals,dtype=np.float64); np_p_exit_type_vals_batch=np.array(batch_p_exit_type_vals,dtype=np.int8)
        np_p_trail_atr_p_vals_batch=np.array(batch_p_trail_atr_p_vals,dtype=np.int32); np_p_trail_atr_m_vals_batch=np.array(batch_p_trail_atr_m_vals,dtype=np.float64); np_p_use_vol_f_vals_batch=np.array(batch_p_use_vol_f_vals,dtype=np.int8)
        np_p_vol_sma_p_vals_batch=np.array(batch_p_vol_sma_p_vals,dtype=np.int32); np_p_use_rsi_f_vals_batch=np.array(batch_p_use_rsi_f_vals,dtype=np.int8); np_p_rsi_p_vals_batch=np.array(batch_p_rsi_p_vals,dtype=np.int32)
        np_p_rsi_l_vals_batch=np.array(batch_p_rsi_l_vals,dtype=np.float64); np_p_rsi_s_vals_batch=np.array(batch_p_rsi_s_vals,dtype=np.float64)
        np_p_adx_t_regime_vals_batch=np.array(batch_p_adx_t_regime_vals,dtype=np.float64); np_p_atr_p_regime_vals_batch=np.array(batch_p_atr_p_regime_vals,dtype=np.float64)
        np_p_time_stop_h_vals_batch=np.array(batch_p_time_stop_h_vals,dtype=np.int32); np_p_profit_trail_vals_batch=np.array(batch_p_profit_trail_vals,dtype=np.float64)
        np_p_max_consec_l_vals_batch=np.array(batch_p_max_consec_l_vals,dtype=np.int32); np_p_cooldown_b_vals_batch=np.array(batch_p_cooldown_b_vals,dtype=np.int32)
        np_p_adx_t_short_vals_batch=np.array(batch_p_adx_t_short_vals,dtype=np.float64); np_p_price_bd_p_vals_batch=np.array(batch_p_price_bd_p_vals,dtype=np.int32); np_p_rsi_mom_t_vals_batch=np.array(batch_p_rsi_mom_t_vals,dtype=np.float64)
        
        np_idx_h1_ema_s_batch=np.array(batch_idx_h1_ema_s,dtype=np.int32); np_idx_h1_ema_l_batch=np.array(batch_idx_h1_ema_l,dtype=np.int32); np_idx_h4_ema_h_batch=np.array(batch_idx_h4_ema_h,dtype=np.int32)
        np_idx_h1_adx_batch=np.array(batch_idx_h1_adx,dtype=np.int32); np_idx_h1_atr_sl_batch=np.array(batch_idx_h1_atr_sl,dtype=np.int32); np_idx_h1_atr_trail_batch=np.array(batch_idx_h1_atr_trail,dtype=np.int32)
        np_idx_h1_vol_sma_batch=np.array(batch_idx_h1_vol_sma,dtype=np.int32); np_idx_h1_rsi_batch=np.array(batch_idx_h1_rsi,dtype=np.int32)
        batch_prep_duration = time.time() - batch_prep_start
        logger.debug(f"배치 {batch_num} Numpy 배열 생성 완료 ({batch_prep_duration:.2f} 초)")

        batch_gpu_transfer_start = time.time()
        d_p_ema_s_vals_batch=cuda.to_device(np_p_ema_s_vals_batch); d_p_ema_l_vals_batch=cuda.to_device(np_p_ema_l_vals_batch); d_p_ema_h_vals_batch=cuda.to_device(np_p_ema_h_vals_batch)
        d_p_adx_p_vals_batch=cuda.to_device(np_p_adx_p_vals_batch); d_p_adx_t_vals_batch=cuda.to_device(np_p_adx_t_vals_batch); d_p_atr_p_sl_vals_batch=cuda.to_device(np_p_atr_p_sl_vals_batch)
        d_p_atr_m_sl_vals_batch=cuda.to_device(np_p_atr_m_sl_vals_batch); d_p_rr_vals_batch=cuda.to_device(np_p_rr_vals_batch); d_p_use_htf_f_vals_batch=cuda.to_device(np_p_use_htf_f_vals_batch)
        d_p_use_adx_f_vals_batch=cuda.to_device(np_p_use_adx_f_vals_batch); d_p_risk_vals_batch=cuda.to_device(np_p_risk_vals_batch); d_p_exit_type_vals_batch=cuda.to_device(np_p_exit_type_vals_batch)
        d_p_trail_atr_p_vals_batch=cuda.to_device(np_p_trail_atr_p_vals_batch); d_p_trail_atr_m_vals_batch=cuda.to_device(np_p_trail_atr_m_vals_batch); d_p_use_vol_f_vals_batch=cuda.to_device(np_p_use_vol_f_vals_batch)
        d_p_vol_sma_p_vals_batch=cuda.to_device(np_p_vol_sma_p_vals_batch); d_p_use_rsi_f_vals_batch=cuda.to_device(np_p_use_rsi_f_vals_batch); d_p_rsi_p_vals_batch=cuda.to_device(np_p_rsi_p_vals_batch)
        d_p_rsi_l_vals_batch=cuda.to_device(np_p_rsi_l_vals_batch); d_p_rsi_s_vals_batch=cuda.to_device(np_p_rsi_s_vals_batch)
        d_p_use_regime_f_vals_batch=cuda.to_device(np_p_use_regime_f_vals_batch)
        d_p_adx_t_regime_vals_batch=cuda.to_device(np_p_adx_t_regime_vals_batch); d_p_atr_p_regime_vals_batch=cuda.to_device(np_p_atr_p_regime_vals_batch)
        d_p_time_stop_h_vals_batch=cuda.to_device(np_p_time_stop_h_vals_batch); d_p_profit_trail_vals_batch=cuda.to_device(np_p_profit_trail_vals_batch)
        d_p_max_consec_l_vals_batch=cuda.to_device(np_p_max_consec_l_vals_batch); d_p_cooldown_b_vals_batch=cuda.to_device(np_p_cooldown_b_vals_batch)
        d_p_adx_t_short_vals_batch=cuda.to_device(np_p_adx_t_short_vals_batch); d_p_price_bd_p_vals_batch=cuda.to_device(np_p_price_bd_p_vals_batch); d_p_rsi_mom_t_vals_batch=cuda.to_device(np_p_rsi_mom_t_vals_batch)

        d_idx_h1_ema_s_batch=cuda.to_device(np_idx_h1_ema_s_batch); d_idx_h1_ema_l_batch=cuda.to_device(np_idx_h1_ema_l_batch); d_idx_h4_ema_h_batch=cuda.to_device(np_idx_h4_ema_h_batch)
        d_idx_h1_adx_batch=cuda.to_device(np_idx_h1_adx_batch); d_idx_h1_atr_sl_batch=cuda.to_device(np_idx_h1_atr_sl_batch); d_idx_h1_atr_trail_batch=cuda.to_device(np_idx_h1_atr_trail_batch)
        d_idx_h1_vol_sma_batch=cuda.to_device(np_idx_h1_vol_sma_batch); d_idx_h1_rsi_batch=cuda.to_device(np_idx_h1_rsi_batch)

        d_out_final_balances_batch = cuda.device_array(current_batch_size, dtype=np.float64)
        d_out_total_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32)
        d_out_win_trades_batch = cuda.device_array(current_batch_size, dtype=np.int32)
        d_out_error_flags_batch = cuda.device_array(current_batch_size, dtype=np.int8)
        d_out_pnl_percentages_batch = cuda.device_array(current_batch_size, dtype=np.float64)
        d_out_profit_factors_batch = cuda.device_array(current_batch_size, dtype=np.float64)
        d_out_max_drawdowns_batch = cuda.device_array(current_batch_size, dtype=np.float64)
        batch_gpu_transfer_duration = time.time() - batch_gpu_transfer_start
        logger.debug(f"배치 {batch_num} GPU 데이터 전송 완료 ({batch_gpu_transfer_duration:.2f} 초)")

        threads_per_block_batch = 128
        blocks_per_grid_batch = (current_batch_size + threads_per_block_batch - 1) // threads_per_block_batch
        logger.info(f"배치 {batch_num} GPU 커널 실행 시작... Blocks: {blocks_per_grid_batch}, Threads: {threads_per_block_batch}")
        batch_kernel_start = time.time()
        run_batch_backtest_gpu_kernel[blocks_per_grid_batch, threads_per_block_batch](
            d_open_p, d_high_p, d_low_p, d_close_p, d_volume, 
            d_h1_indicators, d_h4_indicators, d_hour_of_day_all, d_day_of_week_all,                

            d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch,
            d_p_adx_p_vals_batch, d_p_adx_t_vals_batch,
            d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch,
            d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch,
            d_p_risk_vals_batch, d_p_exit_type_vals_batch,
            d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch,
            d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch,
            d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch, d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch,
            d_p_adx_t_regime_vals_batch, d_p_atr_p_regime_vals_batch,
            d_p_time_stop_h_vals_batch, d_p_profit_trail_vals_batch,
            d_p_max_consec_l_vals_batch, d_p_cooldown_b_vals_batch,
            d_p_adx_t_short_vals_batch, d_p_price_bd_p_vals_batch, d_p_rsi_mom_t_vals_batch,

            d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch,
            d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch, d_idx_h1_atr_trail_batch,
            d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch,

            d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch,
            d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch,

            data_length_global, initial_balance, commission_rate_backtest, slippage_rate_per_trade,
            min_trade_size_btc, quantity_precision_bt,
            d_allowed_hours_bool_global, d_blocked_days_bool_global
        )
        cuda.synchronize() 
        batch_kernel_duration = time.time() - batch_kernel_start
        logger.info(f"배치 {batch_num} GPU 커널 실행 완료 ({batch_kernel_duration:.2f} 초)")

        batch_result_copy_start = time.time()
        h_final_balances_batch = d_out_final_balances_batch.copy_to_host()
        h_total_trades_batch = d_out_total_trades_batch.copy_to_host()
        h_win_trades_batch = d_out_win_trades_batch.copy_to_host()
        h_error_flags_batch = d_out_error_flags_batch.copy_to_host()
        h_pnl_percentages_batch = d_out_pnl_percentages_batch.copy_to_host()
        h_profit_factors_batch = d_out_profit_factors_batch.copy_to_host()
        h_max_drawdowns_batch = d_out_max_drawdowns_batch.copy_to_host()
        batch_result_copy_duration = time.time() - batch_result_copy_start
        logger.debug(f"배치 {batch_num} CPU로 결과 복사 완료 ({batch_result_copy_duration:.2f} 초)")

        for j in range(current_batch_size):
            param_id = current_batch_param_ids[j]
            num_trades_for_combo = h_total_trades_batch[j]

            performance_summary = {
                "param_id": param_id,
                "initial_balance": initial_balance,
                "final_balance": round(h_final_balances_batch[j], 2),
                "total_net_pnl": round(h_final_balances_batch[j] - initial_balance, 2),
                "total_net_pnl_percentage": round(h_pnl_percentages_batch[j], 2),
                "num_trades": num_trades_for_combo,
                "num_wins": h_win_trades_batch[j],
                "num_losses": num_trades_for_combo - h_win_trades_batch[j],
                "win_rate_percentage": round((h_win_trades_batch[j] / num_trades_for_combo) * 100 if num_trades_for_combo > 0 else 0, 2),
                "profit_factor": round(h_profit_factors_batch[j], 2) if h_profit_factors_batch[j] != float('inf') else 'inf', 
                "max_drawdown_percentage": round(h_max_drawdowns_batch[j], 2),
                "error": bool(h_error_flags_batch[j]) 
            }
            all_results_summary.append(performance_summary)

        # 배치별 GPU 메모리 정리
        del d_p_ema_s_vals_batch, d_p_ema_l_vals_batch, d_p_ema_h_vals_batch
        del d_p_adx_p_vals_batch, d_p_adx_t_vals_batch, d_p_atr_p_sl_vals_batch, d_p_atr_m_sl_vals_batch
        del d_p_rr_vals_batch, d_p_use_htf_f_vals_batch, d_p_use_adx_f_vals_batch, d_p_risk_vals_batch, d_p_exit_type_vals_batch
        del d_p_trail_atr_p_vals_batch, d_p_trail_atr_m_vals_batch, d_p_use_vol_f_vals_batch, d_p_vol_sma_p_vals_batch
        del d_p_use_rsi_f_vals_batch, d_p_rsi_p_vals_batch, d_p_rsi_l_vals_batch, d_p_rsi_s_vals_batch
        del d_p_adx_t_regime_vals_batch, d_p_atr_p_regime_vals_batch
        del d_p_time_stop_h_vals_batch, d_p_profit_trail_vals_batch
        del d_p_max_consec_l_vals_batch, d_p_cooldown_b_vals_batch
        del d_p_adx_t_short_vals_batch, d_p_price_bd_p_vals_batch, d_p_rsi_mom_t_vals_batch
        del d_idx_h1_ema_s_batch, d_idx_h1_ema_l_batch, d_idx_h4_ema_h_batch
        del d_idx_h1_adx_batch, d_idx_h1_atr_sl_batch, d_idx_h1_atr_trail_batch
        del d_idx_h1_vol_sma_batch, d_idx_h1_rsi_batch
        del d_out_final_balances_batch, d_out_total_trades_batch, d_out_win_trades_batch, d_out_error_flags_batch
        del d_out_pnl_percentages_batch, d_out_profit_factors_batch, d_out_max_drawdowns_batch
        gc.collect() # 명시적 가비지 컬렉션

        batch_duration = time.time() - batch_start_time
        logger.info(f"--- 배치 {batch_num} 처리 완료 ({batch_duration:.2f} 초) ---")

    num_final_combinations = total_combinations_processed # 총 처리된 조합 수 업데이트
    logger.info(f"총 {num_final_combinations}개의 고유 파라미터 조합 준비 및 처리 완료.")
    prep_duration = time.time() - prep_start_time
    logger.info(f"파라미터 준비 및 처리 완료. 소요 시간: {prep_duration:.2f} 초")

    logger.info("공통 데이터를 GPU로 전송 중...")
    gpu_common_transfer_start = time.time()
    d_open_p=cuda.to_device(open_p_np); d_high_p=cuda.to_device(high_p_np); d_low_p=cuda.to_device(low_p_np); d_close_p=cuda.to_device(close_p_np); d_volume=cuda.to_device(volume_np)
    d_hour_of_day_all=cuda.to_device(hour_of_day_np); d_day_of_week_all=cuda.to_device(day_of_week_np)
    d_allowed_hours_bool_global=cuda.to_device(allowed_hours_bool_np); d_blocked_days_bool_global=cuda.to_device(blocked_days_bool_np)
    d_h1_indicators=cuda.to_device(master_h1_indicators_np)
    d_h4_indicators = cuda.to_device(master_h4_indicators_np) if master_h4_indicators_np is not None else cuda.to_device(np.full_like(master_h1_indicators_np, np.nan)) 

    gpu_common_transfer_duration = time.time() - gpu_common_transfer_start
    logger.info(f"공통 데이터 GPU 전송 완료. 소요 시간: {gpu_common_transfer_duration:.2f} 초")
    del master_h1_indicators_np, master_h4_indicators_np 
    del open_p_np, high_p_np, low_p_np, close_p_np, volume_np
    gc.collect()
        param_id_parts.append(f"Risk{c_id['risk_per_trade_percentage']*100:.1f}%")
        param_id_str = "_".join(map(str, param_id_parts))


        if param_id_str not in seen_ids_final:
            seen_ids_final.add(param_id_str)
            final_param_combos_dicts.append(combo) 
            final_param_ids.append(param_id_str)

            final_p_ema_s_vals.append(combo['ema_short_h1']); final_p_ema_l_vals.append(combo['ema_long_h1']); final_p_ema_h_vals.append(combo['ema_htf'])
            final_p_adx_p_vals.append(combo['adx_period']); final_p_adx_t_vals.append(combo['adx_threshold']); final_p_atr_p_sl_vals.append(combo['atr_period_sl']); final_p_atr_m_sl_vals.append(combo['atr_multiplier_sl'])
            final_p_rr_vals.append(combo['risk_reward_ratio']); final_p_use_htf_f_vals.append(1 if combo.get('use_htf_ema_filter') else 0); final_p_use_adx_f_vals.append(1 if combo.get('use_adx_filter') else 0)
            final_p_risk_vals.append(combo['risk_per_trade_percentage']); final_p_exit_type_vals.append(EXIT_STRATEGY_FIXED_RR_GPU if combo['exit_strategy_type'] == 'FixedRR' else EXIT_STRATEGY_TRAILING_ATR_GPU)
            final_p_trail_atr_p_vals.append(combo['trailing_atr_period']); final_p_trail_atr_m_vals.append(combo['trailing_atr_multiplier'])
            final_p_use_vol_f_vals.append(1 if combo.get('use_volume_filter') else 0); final_p_vol_sma_p_vals.append(combo['volume_sma_period'])
            final_p_use_rsi_f_vals.append(1 if combo.get('use_rsi_filter') else 0); final_p_rsi_p_vals.append(combo['rsi_period']); final_p_rsi_l_vals.append(combo['rsi_threshold_long']); final_p_rsi_s_vals.append(combo['rsi_threshold_short'])

            # Phase 1 Parameters
            final_p_adx_t_regime_vals.append(combo['adx_threshold_regime']); final_p_atr_p_regime_vals.append(combo['atr_percent_threshold_regime'])
            final_p_time_stop_h_vals.append(combo['time_stop_period_hours']); final_p_profit_trail_vals.append(combo['profit_threshold_for_trail'])
            final_p_max_consec_l_vals.append(combo['max_consecutive_losses']); final_p_cooldown_b_vals.append(combo['cooldown_period_bars'])
            final_p_adx_t_short_vals.append(combo['adx_threshold_for_short']); final_p_price_bd_p_vals.append(combo['price_breakdown_period']); final_p_rsi_mom_t_vals.append(combo['rsi_momentum_threshold'])

            final_idx_h1_ema_s.append(indicator_to_idx_map.get(('ema', combo['ema_short_h1']), -1))
            final_idx_h1_ema_l.append(indicator_to_idx_map.get(('ema', combo['ema_long_h1']), -1))
            final_idx_h4_ema_h.append(indicator_to_idx_map.get(('ema', combo['ema_htf']), -1) if combo.get('use_htf_ema_filter') else -1) 
            final_idx_h1_adx.append(indicator_to_idx_map.get(('adx', combo['adx_period']), -1) if combo.get('use_adx_filter') else -1)
            final_idx_h1_atr_sl.append(indicator_to_idx_map.get(('atr', combo['atr_period_sl']), -1))
            final_idx_h1_atr_trail.append(indicator_to_idx_map.get(('atr', combo['trailing_atr_period']), -1) if combo['exit_strategy_type'] == 'TrailingATR' else -1)
            final_idx_h1_vol_sma.append(indicator_to_idx_map.get(('vol_sma', combo['volume_sma_period']), -1) if combo.get('use_volume_filter') else -1)
            final_idx_h1_rsi.append(indicator_to_idx_map.get(('rsi', combo['rsi_period']), -1) if combo.get('use_rsi_filter') else -1)

        if processed_count % log_interval == 0:
            logger.info(f"파라미터 준비 진행: {processed_count}/{total_combinations_estimate} 개 조합 처리됨 ({len(seen_ids_final)} 개 유효 ID)")

    num_final_combinations = total_combinations_processed # 총 처리된 조합 수 업데이트
    logger.info(f"총 {num_final_combinations}개의 고유 파라미터 조합 준비 및 처리 완료.")
    prep_duration = time.time() - prep_start_time
    logger.info(f"파라미터 준비 및 처리 완료. 소요 시간: {prep_duration:.2f} 초")

    logger.info("최종 결과 처리 및 요약 시작...")
    summary_start_time = time.time()

    best_performer = None
    best_metric_val = -float('inf')
    comparison_metric_key = 'total_net_pnl_percentage' 

    for res in all_results_summary:
        if not res["error"] and res.get('num_trades', 0) >= 10: 
            metric = res.get(comparison_metric_key, -float('inf'))
            if isinstance(metric, (int, float)) and metric > best_metric_val:
                best_metric_val = metric
                best_performer = res

    summary_duration = time.time() - summary_start_time
    logger.info(f"결과 처리 및 요약 완료. 소요 시간: {summary_duration:.2f} 초")

    overall_duration = time.time() - overall_start_time
    logger.info(f"총 실행 시간: {timedelta(seconds=overall_duration)}")
    logger.info(f"총 {num_final_combinations}개 조합 테스트 완료.")


    if all_results_summary:
        summary_df = pd.DataFrame(all_results_summary)
        summary_df.rename(columns={
            'param_id': 'Param ID', 'total_net_pnl_percentage': 'PnL %',
            'num_trades': 'Num Trades', 'win_rate_percentage': 'Win Rate %',
            'profit_factor': 'Profit Factor', 'max_drawdown_percentage': 'MDD %'
        }, inplace=True)

        try: 
            summary_df_sort_metric = pd.to_numeric(summary_df['PnL %'], errors='coerce').fillna(-float('inf'))
            summary_df.sort_values(by=summary_df_sort_metric.name, ascending=False, inplace=True, na_position='last')
        except Exception as sort_e:
            logger.warning(f"결과 요약 정렬 오류: {sort_e}. PnL % 문자열 기준으로 정렬 시도.")
            summary_df.sort_values(by='PnL %', ascending=False, inplace=True, key=lambda x: pd.to_numeric(x.astype(str).str.replace('inf', '0'), errors='coerce').fillna(-float('inf')))


        logger.info("\n--- 최종 결과 요약 (상위 20개 또는 전체) ---")
        num_to_display = min(20, len(summary_df))
        cols_to_display = ['Param ID', 'PnL %', 'Num Trades', 'Win Rate %', 'Profit Factor', 'MDD %', 'error']
        display_df = summary_df[cols_to_display].head(num_to_display)
        try:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                logger.info(f"\n{display_df.to_string(index=False)}\n")
        except Exception as e:
            logger.info(f"요약 출력 오류: {e}\n{display_df}")


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

                # --- START: Phase 2 JSON 저장 로직 ---
                is_period_start_str = start_date_str 
                is_period_end_str = end_date_str     

                performance_metrics_for_json = {
                    "PnL_percentage": best_performer.get("total_net_pnl_percentage"),
                    "MDD_percentage": best_performer.get("max_drawdown_percentage"),
                    "win_rate_percentage": best_performer.get("win_rate_percentage"),
                    "num_trades": best_performer.get("num_trades"),
                    "profit_factor": best_performer.get("profit_factor")
                }
                
                # original_params_for_best (superm_extended.py)를 real_M1.py의 TRADING_PARAMS 형식으로 매핑
                mapped_parameters = {
                    'symbol': symbol_backtest, # 백테스팅에 사용된 심볼
                    'interval_primary': interval_primary_bt, # 백테스팅에 사용된 기본 인터벌

                    'ema_short_period': original_params_for_best.get('ema_short_h1'),
                    'ema_long_period': original_params_for_best.get('ema_long_h1'),
                    
                    'rsi_period': original_params_for_best.get('rsi_period'),
                    'rsi_threshold_long': original_params_for_best.get('rsi_threshold_long'),
                    'rsi_threshold_short': original_params_for_best.get('rsi_threshold_short'),
                    
                    'atr_period_sl': original_params_for_best.get('atr_period_sl'),
                    'atr_multiplier_sl': original_params_for_best.get('atr_multiplier_sl'),
                    
                    # risk_reward_ratio는 FixedRR exit_strategy_type일 때만 의미가 있음.
                    # real_M1.py는 현재 FixedRR만 지원하므로, 이 값을 사용.
                    'risk_reward_ratio': original_params_for_best.get('risk_reward_ratio') 
                                         if original_params_for_best.get('exit_strategy_type') == 'FixedRR' 
                                         else param_ranges['risk_reward_ratio'][0], # 기본값 또는 TrailingATR 시 real_M1의 기본값 사용 가정

                    'use_adx_filter': original_params_for_best.get('use_adx_filter', False),
                    'adx_period': original_params_for_best.get('adx_period', param_ranges['adx_period'][0]), # 기본값 사용
                    'adx_threshold': original_params_for_best.get('adx_threshold', param_ranges['adx_threshold'][0]), # 기본값 사용
                    
                    'use_htf_ema_filter': original_params_for_best.get('use_htf_ema_filter', False),
                    # 'ema_htf_period': original_params_for_best.get('ema_htf'), # real_M1.py에 이 키가 없으므로 주석 처리 또는 추가 필요
                    
                    'use_volume_filter': original_params_for_best.get('use_volume_filter', False),
                    # 'volume_sma_period': original_params_for_best.get('volume_sma_period'), # real_M1.py에 이 키가 없으므로 주석 처리 또는 추가 필요

                    'risk_per_trade_percentage': original_params_for_best.get('risk_per_trade_percentage'),
                    
                    # 아래 파라미터들은 real_M1.py의 TRADING_PARAMS에 있지만,
                    # WFA 최적화 대상이 아니거나 superm_extended.py에 없는 값들.
                    # real_M1.py에서 JSON 로드 시 이 값들이 없으면 기본값을 사용하거나,
                    # 여기서 고정된 값을 넣어주거나, real_M1.py에서 별도 관리해야 함.
                    # 여기서는 포함하지 않음. real_M1.py가 스마트하게 처리하도록 기대.
                    # 'leverage_config': 2, 
                    # 'tick_size': price_precision_bt, 
                    # 'min_contract_size': min_trade_size_btc, 
                    # 'quantity_precision': quantity_precision_bt, 
                    # 'price_precision': price_precision_bt,
                    # 'min_trade_value_usdt': 5,
                    # 'db_path': 'trading_data_BTCUSDT.sqlite',

                    # exit_strategy_type 관련 파라미터 (real_M1.py가 현재 FixedRR만 지원)
                    # 만약 WFA가 TrailingATR을 최적으로 선택했더라도, real_M1.py가 이를 지원하지 않으면
                    # 이 정보는 사용되지 않거나, real_M1.py의 수정이 필요함.
                    # 여기서는 일단 정보를 포함시켜둠.
                    'exit_strategy_type': original_params_for_best.get('exit_strategy_type'),
                    'trailing_atr_period': original_params_for_best.get('trailing_atr_period') 
                                           if original_params_for_best.get('exit_strategy_type') == 'TrailingATR' 
                                           else None, # FixedRR일 경우 None 또는 기본값
                    'trailing_atr_multiplier': original_params_for_best.get('trailing_atr_multiplier')
                                               if original_params_for_best.get('exit_strategy_type') == 'TrailingATR'
                                               else None, # FixedRR일 경우 None 또는 기본값
                }
                
                # None 값을 가진 파라미터 제거 (선택적, real_M1.py의 로딩 로직에 따라 결정)
                # mapped_parameters = {k: v for k, v in mapped_parameters.items() if v is not None}


                output_data_for_json = {
                    "optimized_for_is_period_start": is_period_start_str,
                    "optimized_for_is_period_end": is_period_end_str,
                    "is_performance": performance_metrics_for_json,
                    "generated_at": datetime.now().isoformat(), # ISO 8601 형식 시간
                    "parameters": mapped_parameters,
                    "backtest_param_id": best_param_id_found # WFA에서 사용된 고유 ID도 포함
                }

                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 파일 이름에 WFA 기간 정보나 OOS 기간 정보를 포함하는 것이 더 좋을 수 있음 (향후 WFA 구조에 따라)
                # 예: optimized_params_IS_20230101-20240101_OOS_20240102-20240201_20250509_150000.json
                suffix_tag = os.getenv('JSON_TAG')
                base_name = f"optimized_params_{symbol_backtest}_{interval_primary_bt.replace(' ','')}_{timestamp_str}"
                if suffix_tag:
                    output_filename = f"{base_name}_{suffix_tag}.json"
                else:
                    output_filename = f"{base_name}.json"
                
                # 저장할 디렉토리 (예: 'wfa_output_parameters')
                # 이 디렉토리는 클라우드 스토리지와 동기화될 폴더의 하위 경로가 될 수 있음
                output_dir = "wfa_optimized_params_output" 
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    logger.info(f"'{output_dir}' 디렉토리를 생성했습니다.")
                output_filepath = os.path.join(output_dir, output_filename)

                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(output_data_for_json, f, ensure_ascii=False, indent=4, default=_json_default)
                    logger.info(f"최적 파라미터 및 성과를 JSON 파일로 저장했습니다: '{output_filepath}'")
                except Exception as e:
                    logger.error(f"최적 파라미터 JSON 파일 저장 중 오류 발생: {e}", exc_info=True)
                # --- END: Phase 2 JSON 저장 로직 ---
            
            logger.info("상세 거래 내역은 저장되지 않았습니다. 필요시 최고 성과 파라미터로 개별 백테스트를 실행하세요.") # 이 메시지는 JSON 저장 후에도 유효
        else:
            logger.info(f"\n최소 거래 횟수(10회) 및 오류 없음 조건을 만족하는 최고 성과 조합을 찾지 못했습니다.")
    else:
        logger.info("\n테스트 결과가 없습니다.")


    try:
        del d_open_p, d_high_p, d_low_p, d_close_p, d_volume, d_h1_indicators, d_h4_indicators
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        logger.info("공통 GPU 데이터 및 CuPy 메모리 풀 정리 시도 완료.")
    except Exception as e:
        logger.warning(f"공통 GPU 메모리 정리 중 오류: {e}")

    logger.info("백테스터 V5.11 (최적화 파라미터 범위) 실행 완료.")
