# -*- coding: utf-8 -*-
# 실매매 봇 Phase 1 안정화 버전 (sil_phase1_v1.py)
# 원본: sil.py (사용자 제공)
# 개선: TRADING_PARAMS 중앙 관리, 로깅/오류 처리 검토, Phase 1 목표 주석 추가, 이메일 알림 기능, SQLite 데이터 저장 추가
# 주의: 이 코드는 실제 자금 거래를 위한 기본 구조이며,
# 반드시 페이퍼 트레이딩 환경에서 충분한 테스트와 검증을 거친 후 사용해야 합니다.
# 모든 투자 결정과 그 결과는 사용자 본인의 책임입니다.

import os
import schedule
import time
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import * # ENUM 사용을 위해 import 확인
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import logging
from datetime import datetime, timedelta, timezone
import json
import smtplib 
from email.mime.text import MIMEText
import sqlite3 # SQLite 데이터베이스 사용을 위해 추가

from src.filters.regime_filter import is_favorable_regime
from src.policy.time_windows import is_trade_allowed_by_time
from src.position.sizing import on_trade_closed, can_open_new_trade, MAX_CONSECUTIVE_LOSSES, COOLDOWN_PERIOD_BARS
from src.exits.trailing import update_and_check_atr_trailing_stop, check_fixed_stop_loss

# --- 로깅 설정 ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
log_file_path = 'live_trading_bot_phase1.log' 
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO) 
if not logger.hasHandlers(): 
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
logger.info(f"로그 파일 위치: {os.path.abspath(log_file_path)}")

# --- .env 파일 로드 ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env 파일 로드 완료.")
else:
    logger.warning("경고: .env 파일을 찾을 수 없습니다. API 키 및 이메일 정보가 환경 변수에 설정되어 있는지 확인하세요.")

# --- API 키 설정 ---
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_SECRET_KEY')

# --- 이메일 알림 설정 로드 ---
EMAIL_SENDER_ADDRESS = os.getenv('EMAIL_SENDER_ADDRESS')
EMAIL_SENDER_PASSWORD = os.getenv('EMAIL_SENDER_PASSWORD')
EMAIL_RECEIVER_ADDRESS = os.getenv('EMAIL_RECEIVER_ADDRESS')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')

EMAIL_ENABLED = False
# API 키 확인 후 이메일 설정 확인 (API 키 없으면 이메일 발송 의미 없음)
if API_KEY and API_SECRET:
    if EMAIL_SENDER_ADDRESS and EMAIL_SENDER_PASSWORD and EMAIL_RECEIVER_ADDRESS and SMTP_SERVER and SMTP_PORT:
        try:
            SMTP_PORT = int(SMTP_PORT) 
            EMAIL_ENABLED = True
            logger.info("이메일 알림 설정이 성공적으로 로드되었습니다.")
        except ValueError:
            logger.error("오류: SMTP_PORT가 올바른 숫자 형식이 아닙니다. 이메일 발송이 비활성화됩니다.")
    else:
        logger.warning("이메일 알림 설정이 .env 파일에 충분히 제공되지 않았습니다. (EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD, EMAIL_RECEIVER_ADDRESS, SMTP_SERVER, SMTP_PORT 모두 필요). 이메일 발송을 건너뜁니다.")
else:
    logger.error("오류: API 키와 시크릿 키를 .env 파일 또는 환경 변수에 설정해주세요! 프로그램을 종료합니다.")
    # API 키가 없을 때 이메일 발송 시도하면 EMAIL_ENABLED가 False일 것이므로, 여기서는 exit만.
    exit()


def send_email_notification(subject, body):
    """지정된 제목과 내용으로 이메일을 발송합니다."""
    if not EMAIL_ENABLED:
        logger.info(f"이메일 기능 비활성화됨. 제목: {subject}, 내용: {body[:100]}...") 
        return

    msg = MIMEText(body)
    msg['Subject'] = f"[자동매매봇 알림] {subject}" 
    msg['From'] = EMAIL_SENDER_ADDRESS
    msg['To'] = EMAIL_RECEIVER_ADDRESS

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls() 
            smtp.login(EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD)
            smtp.send_message(msg)
            logger.info(f"이메일 발송 성공: 받는사람={EMAIL_RECEIVER_ADDRESS}, 제목='{msg['Subject']}'")
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP 인증 오류: 이메일 주소 또는 비밀번호를 확인하세요.")
    except Exception as e:
        logger.error(f"이메일 발송 중 오류 발생: {e}", exc_info=True)


# --- 바이낸스 클라이언트 초기화 ---
USE_TESTNET = False # <--- 실매매 시 False, 테스트넷에서 테스트할 때는 True로 설정!
client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
logger.info(f"바이낸스 클라이언트 초기화 완료. (테스트넷 사용: {USE_TESTNET})")
if not USE_TESTNET:
    logger.warning("*" * 30); logger.warning("****** 경고: 실제 매매 계정으로 실행됩니다! ******"); logger.warning("*" * 30)
else:
    logger.info("테스트넷 계정으로 실행됩니다. 실제 자금이 사용되지 않습니다.")

# --- 거래 파라미터 중앙 관리 (TRADING_PARAMS) ---
TRADING_PARAMS = {
    'symbol': 'BTCUSDT',
    'interval_primary': Client.KLINE_INTERVAL_1HOUR,
    'data_fetch_limit_primary': 200, 
    'ema_short_period': 19, 'ema_long_period': 20,
    'rsi_period': 24, 'rsi_threshold_long': 50, 'rsi_threshold_short': 47,
    'atr_period_sl': 12, 'atr_multiplier_sl': 2.8, 'risk_reward_ratio': 2.8,
    'use_adx_filter': False, 'adx_period': 14, 'adx_threshold': 20,
    'use_htf_ema_filter': False, 'use_volume_filter': False,
    'risk_per_trade_percentage': 0.022, 'leverage_config': 2,
    'tick_size': 0.01, 'min_contract_size': 0.001, 
    'quantity_precision': 3, 'price_precision': 2,
    'min_trade_value_usdt': 5,
    # Phase 1 Quick Wins Parameters
    'adx_threshold_regime': 25.0, # For regime_filter.py
    'atr_percent_threshold_regime': 2.0, # For regime_filter.py
    'allowed_hours': {10, 12, 14, 17}, # For time_windows.py
    'blocked_days': {'Monday'},
    'time_stop_period_hours': 48, # For hybrid liquidation
    'profit_threshold_for_trail': 1.0, # For hybrid liquidation
    'trailing_atr_period': 14, # For hybrid liquidation
    'trailing_atr_multiplier': 2.5, # For hybrid liquidation
    'max_consecutive_losses': 4, # For circuit breaker (sizing.py)
    'cooldown_period_bars': 24, # For circuit breaker (sizing.py)
    # Short Position Redesign Parameters
    'adx_threshold_for_short': 25.0,
    'price_breakdown_period': 5,
    'rsi_momentum_threshold': 45.0,
    'db_path': 'trading_data_BTCUSDT.sqlite', # SQLite DB 파일 경로
    # 'klines_table_name'은 아래에서 동적으로 생성하여 추가
}
# 테이블명 동적 생성 (TRADING_PARAMS 초기화 후)
TRADING_PARAMS['klines_table_name'] = f"klines_{TRADING_PARAMS['interval_primary'].replace(' ','')}_{TRADING_PARAMS['symbol'].lower()}"


# --- 지표 컬럼명 설정 ---
ema_short_col = f'EMA_{TRADING_PARAMS["ema_short_period"]}'
ema_long_col = f'EMA_{TRADING_PARAMS["ema_long_period"]}'
rsi_col = f'RSI_{TRADING_PARAMS["rsi_period"]}'
atr_col = f'ATRr_{TRADING_PARAMS["atr_period_sl"]}'

# --- 전역 변수 ---
POSITION_STATE_FILE = f'position_state_{TRADING_PARAMS["symbol"]}_ema_cross_live_phase1.json'
current_position_side = 'None'; current_position_quantity = 0.0; entry_price = 0.0
stop_loss_price = 0.0; take_profit_price = 0.0; last_signal_check_time = None; entry_atr = 0.0

def initialize_exchange_info():
    global TRADING_PARAMS
    try:
        logger.info(f"{TRADING_PARAMS['symbol']} 거래소 정보 조회 시도...")
        exchange_info = client.futures_exchange_info()
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == TRADING_PARAMS['symbol']:
                TRADING_PARAMS['quantity_precision'] = s_info['quantityPrecision']
                TRADING_PARAMS['price_precision'] = s_info['pricePrecision']
                for f_info in s_info['filters']:
                    if f_info['filterType'] == 'PRICE_FILTER':
                        TRADING_PARAMS['tick_size'] = float(f_info['tickSize'])
                    elif f_info['filterType'] == 'LOT_SIZE':
                        TRADING_PARAMS['min_contract_size'] = float(f_info['minQty'])
                logger.info(f"{TRADING_PARAMS['symbol']} 정보 업데이트 완료: TickSize={TRADING_PARAMS['tick_size']}, MinQty={TRADING_PARAMS['min_contract_size']}, QtyPrecision={TRADING_PARAMS['quantity_precision']}, PricePrecision={TRADING_PARAMS['price_precision']}")
                return True
        logger.error(f"{TRADING_PARAMS['symbol']} 정보를 찾을 수 없습니다.")
        return False
    except BinanceAPIException as e: logger.error(f"API 오류 (거래소 정보 조회): {e}"); return False
    except Exception as e: logger.error(f"알 수 없는 오류 (거래소 정보 조회): {e}"); return False

def save_position_state():
    state = {
        'current_position_side': current_position_side, 'current_position_quantity': current_position_quantity,
        'entry_price': entry_price, 'stop_loss_price': stop_loss_price, 'take_profit_price': take_profit_price,
        'last_signal_check_time': last_signal_check_time.isoformat() if last_signal_check_time else None,
        'entry_atr': entry_atr
    }
    try:
        with open(POSITION_STATE_FILE, 'w') as f: json.dump(state, f, indent=4)
        logger.debug(f"포지션 상태 저장 완료: {POSITION_STATE_FILE}")
    except Exception as e: logger.error(f"포지션 상태 저장 실패: {e}")

def load_position_state():
    global current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price, last_signal_check_time, entry_atr
    try:
        if os.path.exists(POSITION_STATE_FILE):
            with open(POSITION_STATE_FILE, 'r') as f: state = json.load(f)
            current_position_side = state.get('current_position_side', 'None')
            current_position_quantity = state.get('current_position_quantity', 0.0)
            entry_price = state.get('entry_price', 0.0)
            stop_loss_price = state.get('stop_loss_price', 0.0)
            take_profit_price = state.get('take_profit_price', 0.0)
            last_signal_check_time_iso = state.get('last_signal_check_time')
            if last_signal_check_time_iso:
                ts = pd.Timestamp(last_signal_check_time_iso)
                last_signal_check_time = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
            else: last_signal_check_time = None
            entry_atr = state.get('entry_atr', 0.0)
            logger.info(f"포지션 상태 불러오기 완료: Side={current_position_side}, Qty={current_position_quantity}, Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}, LastCheck={last_signal_check_time}, EntryATR={entry_atr}")
        else: logger.info("저장된 포지션 상태 파일 없음. 초기 상태로 시작.")
    except Exception as e:
        logger.error(f"포지션 상태 불러오기 실패: {e}. 초기 상태로 시작.")
        current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price, last_signal_check_time, entry_atr = 'None', 0.0, 0.0, 0.0, 0.0, None, 0.0

def get_historical_data_primary(symbol_param, interval_param, limit_param):
    try:
        klines = client.futures_klines(symbol=symbol_param, interval=interval_param, limit=limit_param)
        if not klines: logger.warning(f"{interval_param} 데이터 수신 실패."); return None
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        df.set_index('Open time', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
        df.sort_index(inplace=True); return df
    except BinanceAPIException as e: logger.error(f"API 오류 ({interval_param} 데이터): {e}"); return None
    except Exception as e: logger.error(f"알 수 없는 오류 ({interval_param} 데이터): {e}"); return None

def calculate_indicators(df):
    if df is None or df.empty: logger.warning("지표 계산용 DF 비어있음."); return None
    try:
        df.ta.ema(length=TRADING_PARAMS['ema_short_period'], append=True, col_names=(ema_short_col,))
        df.ta.ema(length=TRADING_PARAMS['ema_long_period'], append=True, col_names=(ema_long_col,))
        df.ta.rsi(length=TRADING_PARAMS['rsi_period'], append=True, col_names=(rsi_col,))
        df.ta.atr(length=TRADING_PARAMS['atr_period_sl'], append=True, col_names=(atr_col,))
        if TRADING_PARAMS['use_adx_filter']: df.ta.adx(length=TRADING_PARAMS['adx_period'], append=True)
        required_cols = [ema_short_col, ema_long_col, rsi_col, atr_col]
        if any(col not in df.columns or df[col].isnull().all() for col in required_cols):
            logger.error(f"필수 지표 컬럼 누락/NaN: {required_cols}. 현재 컬럼: {df.columns.tolist()}"); return None
        return df
    except Exception as e: logger.error(f"지표 계산 중 오류: {e}", exc_info=True); return None

def check_ema_crossover_signal(df):
    min_candles = max(TRADING_PARAMS['ema_long_period'], TRADING_PARAMS['rsi_period'], TRADING_PARAMS['atr_period_sl']) + 5
    if df is None or len(df) < min_candles: logger.warning(f"신호 확인용 데이터 부족 ({len(df) if df is not None else 0} < {min_candles})."); return 0, None, None
    if len(df) < 2: logger.warning("신호 확인에 최소 2캔들 필요."); return 0, None, None
    last, prev = df.iloc[-1], df.iloc[-2]
    if not all(col in last.index and pd.notna(last[col]) and pd.notna(prev[col]) for col in [ema_short_col, ema_long_col, rsi_col, atr_col]):
        logger.warning("최신/이전 캔들 지표 값 부족/NaN."); return 0, None, None
    current_close, current_atr = last['Close'], last[atr_col]
    if current_atr <= 0 or pd.isna(current_atr): logger.warning(f"ATR 값({current_atr}) 유효하지 않음."); return 0, None, None
    
    long_cross = prev[ema_short_col] < prev[ema_long_col] and last[ema_short_col] > last[ema_long_col]
    short_cross = prev[ema_short_col] > prev[ema_long_col] and last[ema_short_col] < last[ema_long_col]
    signal, sl_est, tp_est = 0, None, None

    if long_cross and last[rsi_col] > TRADING_PARAMS['rsi_threshold_long']:
        signal = 1
        sl_dist = current_atr * TRADING_PARAMS['atr_multiplier_sl']
        sl_est = current_close - sl_dist
        tp_est = current_close + (sl_dist * TRADING_PARAMS['risk_reward_ratio'])
        logger.info(f"매수 신호: EMA크로스, RSI={last[rsi_col]:.2f}>{TRADING_PARAMS['rsi_threshold_long']}. SL_Est={format_price(sl_est)}, TP_Est={format_price(tp_est)}")
    elif short_cross: # EMA cross for short
        # Apply confluence filter for short entry
        if check_short_entry_conditions(df):
            signal = -1
            sl_dist = current_atr * TRADING_PARAMS['atr_multiplier_sl']
            sl_est = current_close + sl_dist
            tp_est = current_close - (sl_dist * TRADING_PARAMS['risk_reward_ratio'])
            logger.info(f"매도 신호: EMA크로스 및 컨플루언스 조건 충족. SL_Est={format_price(sl_est)}, TP_Est={format_price(tp_est)}")
        else:
            logger.info("매도 신호: EMA크로스 발생했으나 컨플루언스 조건 불충족. 진입 안 함.")
    
    if signal != 0 and (sl_est is None or tp_est is None or pd.isna(sl_est) or pd.isna(tp_est) or \
       (signal == 1 and (sl_est >= current_close or tp_est <= current_close or sl_est >= tp_est)) or \
       (signal == -1 and (sl_est <= current_close or tp_est >= current_close or sl_est <= tp_est))):
        logger.warning(f"신호 발생했으나 SL/TP 유효하지 않음. Signal={signal}, Entry={current_close}, SL_Est={sl_est}, TP_Est={tp_est}. 신호 무시.")
        return 0, None, None
    return signal, sl_est, tp_est

def check_short_entry_conditions(df):
    # Ensure required columns exist
    required_cols = [ema_short_col, ema_long_col, 'ADX', 'Close', 'Low', rsi_col]
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"check_short_entry_conditions: 필수 컬럼 누락: {required_cols}")
        return False

    last_candle = df.iloc[-1]
    # Condition 1: Trend Structure (fast EMA below slow EMA)
    is_trend_bearish = last_candle[ema_short_col] < last_candle[ema_long_col]

    # Condition 2: Trend Strength (ADX > threshold for Short)
    is_trend_strong = last_candle['ADX'] > TRADING_PARAMS['adx_threshold_for_short']

    # Condition 3: Price Action Confirmation (recent low broken)
    # Need enough historical data for min() calculation
    if len(df) < TRADING_PARAMS['price_breakdown_period'] + 1:
        logger.warning(f"check_short_entry_conditions: 가격 돌파 확인을 위한 데이터 부족 ({len(df)} < {TRADING_PARAMS['price_breakdown_period'] + 1})")
        return False
    
    # Get the 'Low' prices for the last 'price_breakdown_period' candles excluding the current one
    recent_lows = df['Low'].iloc[-(TRADING_PARAMS['price_breakdown_period'] + 1):-1]
    is_price_breakdown = last_candle['Close'] < recent_lows.min()

    # Condition 4 (Optional): Momentum Oscillator (RSI < threshold)
    is_momentum_bearish = last_candle[rsi_col] < TRADING_PARAMS['rsi_momentum_threshold']

    if is_trend_bearish and is_trend_strong and is_price_breakdown and is_momentum_bearish:
        logger.info("숏 진입 조건 충족: 추세 하락, ADX 강세, 가격 돌파, RSI 약세.")
        return True
    return False

def get_account_balance(asset='USDT'):
    try:
        bal = client.futures_account_balance()
        for b in bal:
            if b['asset'] == asset: return float(b['availableBalance'])
        logger.warning(f"선물 계정 {asset} 잔고 없음."); return 0.0
    except Exception as e: logger.error(f"API 오류 (잔고 조회): {e}"); return 0.0

def calculate_position_size(entry_p, sl_p, balance_val):
    if not all(map(pd.notna, [entry_p, sl_p, balance_val])) or \
       not all(map(lambda x: isinstance(x, (int, float)) and x > 0, [entry_p, balance_val])) or \
       not isinstance(sl_p, (int, float)) or pd.isna(sl_p):
        logger.error(f"포지션 크기 계산 입력값 오류: E={entry_p}, SL={sl_p}, Bal={balance_val}"); return 0.0
    sl_dist = abs(entry_p - sl_p)
    if sl_dist <= TRADING_PARAMS['tick_size'] * 0.1: logger.error(f"SL distance 너무 작음 ({sl_dist})."); return 0.0
    risk_amt = balance_val * TRADING_PARAMS['risk_per_trade_percentage']
    qty_raw = risk_amt / sl_dist
    factor = 10 ** TRADING_PARAMS['quantity_precision']
    qty_adj = math.floor(qty_raw * factor) / factor
    logger.info(f"포지션 크기: Bal={balance_val:.2f}, RiskAmt={risk_amt:.2f}, SL_Dist={sl_dist} -> RawQty={qty_raw:.8f} -> AdjQty={qty_adj}")
    notional = qty_adj * entry_p
    if notional < TRADING_PARAMS['min_trade_value_usdt']: logger.warning(f"계산된 명목가치({notional:.2f}) < 최소주문금액({TRADING_PARAMS['min_trade_value_usdt']})."); return 0.0
    if qty_adj < TRADING_PARAMS['min_contract_size']: logger.warning(f"계산된 수량({qty_adj}) < 최소주문수량({TRADING_PARAMS['min_contract_size']})."); return 0.0
    return qty_adj

def format_price(price_val):
    if price_val is None or pd.isna(price_val) or TRADING_PARAMS['tick_size'] == 0: return f"{float(price_val):.{TRADING_PARAMS['price_precision']}f}" if price_val is not None else None
    return f"{round(float(price_val) / TRADING_PARAMS['tick_size']) * TRADING_PARAMS['tick_size']:.{TRADING_PARAMS['price_precision']}f}"

def format_quantity(quantity_val):
    if quantity_val is None or pd.isna(quantity_val): return None
    return f"{float(quantity_val):.{TRADING_PARAMS['quantity_precision']}f}"

def place_market_order(symbol_p, side_p, qty_p):
    try:
        fmt_qty = format_quantity(qty_p)
        if fmt_qty is None or float(fmt_qty) < TRADING_PARAMS['min_contract_size']: logger.error(f"시장가 주문 수량({fmt_qty}) 유효하지 않음."); return None
        logger.info(f"시장가 주문: {symbol_p}, {side_p}, Qty: {fmt_qty}")
        order = client.futures_create_order(symbol=symbol_p, side=side_p, type=FUTURE_ORDER_TYPE_MARKET, quantity=fmt_qty)
        logger.info(f"시장가 주문 성공: {order}"); return order
    except Exception as e: logger.error(f"시장가 주문 오류: {e}", exc_info=True); send_email_notification(f"봇 오류: 시장가 주문 실패", f"심볼: {symbol_p}, Side: {side_p}, Qty: {fmt_qty}\n오류: {e}"); return None

def place_stop_loss_order(symbol_p, side_p, qty_p, stop_p):
    try:
        fmt_qty, fmt_stop_p = format_quantity(qty_p), format_price(stop_p)
        if fmt_qty is None or float(fmt_qty) < TRADING_PARAMS['min_contract_size'] or fmt_stop_p is None: logger.error(f"SL 주문 수량/가격 유효하지 않음 Qty={fmt_qty}, StopP={fmt_stop_p}."); return None
        logger.info(f"SL 주문: {symbol_p}, {side_p}, Qty: {fmt_qty}, StopPrice: {fmt_stop_p}")
        order = client.futures_create_order(symbol=symbol_p, side=side_p, type=FUTURE_ORDER_TYPE_STOP_MARKET, quantity=fmt_qty, stopPrice=fmt_stop_p, reduceOnly=True)
        logger.info(f"SL 주문 성공: {order}"); return order
    except Exception as e: logger.error(f"SL 주문 오류: {e}", exc_info=True); send_email_notification(f"봇 오류: SL 주문 실패", f"심볼: {symbol_p}, Side: {side_p}, Qty: {fmt_qty}, StopP: {fmt_stop_p}\n오류: {e}"); return None

def place_take_profit_order(symbol_p, side_p, qty_p, stop_p):
    try:
        fmt_qty, fmt_stop_p = format_quantity(qty_p), format_price(stop_p)
        if fmt_qty is None or float(fmt_qty) < TRADING_PARAMS['min_contract_size'] or fmt_stop_p is None: logger.error(f"TP 주문 수량/가격 유효하지 않음 Qty={fmt_qty}, StopP={fmt_stop_p}."); return None
        logger.info(f"TP 주문: {symbol_p}, {side_p}, Qty: {fmt_qty}, StopPrice: {fmt_stop_p}")
        order = client.futures_create_order(symbol=symbol_p, side=side_p, type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, quantity=fmt_qty, stopPrice=fmt_stop_p, reduceOnly=True)
        logger.info(f"TP 주문 성공: {order}"); return order
    except Exception as e: logger.error(f"TP 주문 오류: {e}", exc_info=True); send_email_notification(f"봇 오류: TP 주문 실패", f"심볼: {symbol_p}, Side: {side_p}, Qty: {fmt_qty}, StopP: {fmt_stop_p}\n오류: {e}"); return None

def cancel_all_open_orders(symbol_p):
    try:
        orders = client.futures_get_open_orders(symbol=symbol_p)
        if not orders: logger.info(f"{symbol_p} 미체결 주문 없음."); return True
        logger.info(f"{symbol_p} 미체결 주문 {len(orders)}개 취소 시도...")
        for o in orders:
            try: client.futures_cancel_order(symbol=symbol_p, orderId=o['orderId']); logger.info(f"주문 ID {o['orderId']} 취소됨.")
            except Exception as e_cancel: logger.error(f"주문 ID {o['orderId']} 취소 중 오류: {e_cancel}")
        logger.info(f"{symbol_p} 모든 미체결 주문 취소 시도 완료."); return True 
    except Exception as e: logger.error(f"주문 취소 중 오류: {e}", exc_info=True); return False

def get_position_info(symbol_p):
    try:
        pos = client.futures_position_information(symbol=symbol_p)
        if pos:
            for p in pos:
                if p['symbol'] == symbol_p and float(p.get('positionAmt', 0.0)) != 0: return p
        logger.info(f"{symbol_p} 현재 열린 포지션 없음."); return None
    except BinanceAPIException as e:
        if e.code == -2013: logger.info(f"{symbol_p} 포지션 없음 (API Code: {e.code})"); return None
        logger.error(f"API 오류 (포지션 정보): {e}"); send_email_notification(f"봇 API 오류: 포지션 정보 조회 실패", f"오류: {e}"); return None
    except Exception as e: logger.error(f"알 수 없는 오류 (포지션 정보): {e}", exc_info=True); send_email_notification(f"봇 알 수 없는 오류: 포지션 정보 조회 실패", f"오류: {e}"); return None

def calculate_current_profit(current_price):
    if current_position_side == 'Long':
        return (current_price - entry_price) * current_position_quantity
    elif current_position_side == 'Short':
        return (entry_price - current_price) * current_position_quantity
    return 0.0

def exit_position(reason):
    global current_position_side, current_position_quantity
    logger.info(f"포지션 청산 시작 (사유: {reason})...")
    if current_position_side != 'None' and current_position_quantity > 0:
        order_side = SIDE_SELL if current_position_side == 'Long' else SIDE_BUY
        place_market_order(TRADING_PARAMS['symbol'], order_side, current_position_quantity)
    reset_position_state()
    send_email_notification(f"포지션 청산 ({TRADING_PARAMS['symbol']})", f"사유: {reason}\n청산된 포지션: {current_position_side}\n수량: {current_position_quantity}\n진입가: {entry_price}")

def reset_position_state(save=True):
    global current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price
    logger.info("포지션 상태 초기화 중...")
    if not cancel_all_open_orders(TRADING_PARAMS['symbol']): logger.error("미체결 주문 취소 실패! 수동 확인 필요!")
    current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price = 'None', 0.0, 0.0, 0.0, 0.0
    if save: save_position_state()


def check_position_status_and_sync():
    global current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price
    logger.info("실제 포지션 상태 확인 및 동기화...")
    pos_info = get_position_info(TRADING_PARAMS['symbol'])
    if pos_info:
        api_qty, api_entry = float(pos_info.get('positionAmt',0.0)), float(pos_info.get('entryPrice',0.0))
        api_side = 'Long' if api_qty > 0 else ('Short' if api_qty < 0 else 'None')
        api_qty_abs = abs(api_qty)
        logger.info(f"API 포지션: Side={api_side}, Qty={api_qty_abs}, Entry={api_entry}")
        if current_position_side == 'None':
            logger.warning("로컬 상태 'None', API 포지션 존재. API 기준으로 복구 시도.")
            current_position_side, current_position_quantity, entry_price = api_side, api_qty_abs, api_entry
            if stop_loss_price == 0.0 or take_profit_price == 0.0: logger.warning("복구된 포지션 SL/TP 정보 없음. 재설정 필요.")
            save_position_state()
        elif current_position_side != api_side or abs(current_position_quantity - api_qty_abs) > (TRADING_PARAMS['min_contract_size'] * 0.01):
            logger.error(f"로컬/API 포지션 불일치! 로컬:{current_position_side} {current_position_quantity}@{entry_price}, API:{api_side} {api_qty_abs}@{api_entry}. 리셋 필요!"); send_email_notification(f"봇 긴급 오류: 포지션 불일치", f"로컬: {current_position_side} {current_position_quantity} @ {entry_price}\nAPI: {api_side} {api_qty_abs} @ {api_entry}\n포지션 리셋 및 수동 확인 필요!"); reset_position_state()
        else:
            logger.info("로컬/API 포지션 일치.")
            if abs(entry_price - api_entry) > TRADING_PARAMS['tick_size']: logger.info(f"API 진입가({api_entry})와 로컬 진입가({entry_price}) 차이. API 기준으로 업데이트."); entry_price = api_entry; save_position_state()
    else: 
        logger.info("API 조회 결과, 현재 열린 포지션 없음.")
        if current_position_side != 'None': logger.warning(f"로컬 상태는 {current_position_side} 포지션이나 API에 없음. 리셋."); reset_position_state()
        else: logger.info("로컬/API 모두 포지션 없음 일치.")

def manage_existing_position_sl_tp():
    global stop_loss_price, take_profit_price
    if current_position_side == 'None': return
    logger.info(f"기존 포지션({current_position_side} {current_position_quantity}@{entry_price}) SL/TP 관리...")
    if not (pd.notna(stop_loss_price) and stop_loss_price > 0 and pd.notna(take_profit_price) and take_profit_price > 0):
        logger.warning("포지션 있으나 SL/TP 가격 유효하지 않음. 재계산/재설정 시도.")
        df_atr = get_historical_data_primary(TRADING_PARAMS['symbol'], TRADING_PARAMS['interval_primary'], TRADING_PARAMS['atr_period_sl'] + 5)
        if df_atr is None or df_atr.empty: logger.error("SL/TP 재계산용 ATR 데이터 실패."); return
        df_atr.ta.atr(length=TRADING_PARAMS['atr_period_sl'], append=True, col_names=(atr_col,))
        if atr_col not in df_atr.columns or pd.isna(df_atr.iloc[-1][atr_col]) or df_atr.iloc[-1][atr_col] <=0: logger.error("SL/TP 재계산용 ATR 값 유효하지 않음."); return
        last_atr = df_atr.iloc[-1][atr_col]; sl_dist = last_atr * TRADING_PARAMS['atr_multiplier_sl']
        current_market_p = df_atr.iloc[-1]['Close']
        if current_position_side == 'Long': stop_loss_price, take_profit_price = entry_price - sl_dist, entry_price + (sl_dist * TRADING_PARAMS['risk_reward_ratio'])
        else: stop_loss_price, take_profit_price = entry_price + sl_dist, entry_price - (sl_dist * TRADING_PARAMS['risk_reward_ratio'])
        if (current_position_side == 'Long' and (stop_loss_price >= current_market_p or take_profit_price <= current_market_p or stop_loss_price >= take_profit_price)) or \
           (current_position_side == 'Short' and (stop_loss_price <= current_market_p or take_profit_price >= current_market_p or stop_loss_price <= take_profit_price)) or \
           pd.isna(stop_loss_price) or pd.isna(take_profit_price): logger.error(f"SL/TP 재계산 후 유효하지 않음. SL={stop_loss_price}, TP={take_profit_price}. 청산 고려!"); send_email_notification(f"봇 오류: SL/TP 재계산 실패", f"SL/TP 재계산 후 가격이 유효하지 않습니다. SL={stop_loss_price}, TP={take_profit_price}. 수동 확인 필요."); return 
        logger.info(f"SL/TP 재계산: New SL={format_price(stop_loss_price)}, New TP={format_price(take_profit_price)}"); save_position_state()

    try:
        orders = client.futures_get_open_orders(symbol=TRADING_PARAMS['symbol'])
        has_sl, has_tp = False, False
        sl_tp_side = SIDE_SELL if current_position_side == 'Long' else SIDE_BUY
        for o in orders:
            if abs(float(o['origQty']) - current_position_quantity) < (TRADING_PARAMS['min_contract_size'] * 0.01):
                o_stop_p_fmt, sl_p_fmt, tp_p_fmt = format_price(float(o['stopPrice'])), format_price(stop_loss_price), format_price(take_profit_price)
                if o['type'] == 'STOP_MARKET' and o['side'] == sl_tp_side and o_stop_p_fmt == sl_p_fmt: has_sl = True
                elif o['type'] == 'TAKE_PROFIT_MARKET' and o['side'] == sl_tp_side and o_stop_p_fmt == tp_p_fmt: has_tp = True
        if not has_sl: logger.warning(f"활성 SL 주문 없음/불일치. 재설정 (SL: {format_price(stop_loss_price)})..."); place_stop_loss_order(TRADING_PARAMS['symbol'], sl_tp_side, current_position_quantity, stop_loss_price)
        else: logger.info("활성 SL 주문 확인.")
        if not has_tp: logger.warning(f"활성 TP 주문 없음/불일치. 재설정 (TP: {format_price(take_profit_price)})..."); place_take_profit_order(TRADING_PARAMS['symbol'], sl_tp_side, current_position_quantity, take_profit_price)
        else: logger.info("활성 TP 주문 확인.")
    except Exception as e: logger.error(f"미체결 SL/TP 주문 관리 중 오류: {e}", exc_info=True)

# --- SQLite 데이터 저장 함수 ---
def save_data_to_sqlite(df, db_path, table_name):
    """DataFrame을 SQLite 데이터베이스에 저장합니다. (중복 방지 포함)"""
    if df is None or df.empty:
        logger.info("저장할 데이터가 없습니다 (DataFrame 비어 있음).")
        return
    try:
        with sqlite3.connect(db_path) as conn:
            df_to_save = df.reset_index()
            df_to_save['Open time'] = df_to_save['Open time'].dt.tz_localize(None) 
            try:
                # 테이블 존재 시, 마지막 시간 이후 데이터만 필터링하여 추가하는 것이 더 효율적일 수 있음
                # 여기서는 UNIQUE 제약조건을 가정하고 INSERT OR IGNORE를 사용하거나,
                # pandas의 to_sql을 사용하되, 중복을 수동으로 제거하는 방식을 사용.
                # 'Open time'을 문자열로 변환하여 기본키로 사용하고, UNIQUE 제약조건을 걸 수 있음.
                # 여기서는 간단하게 기존 데이터와 비교하여 없는 것만 추가.
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    "Open time" TIMESTAMP PRIMARY KEY, 
                    "Open" REAL, "High" REAL, "Low" REAL, "Close" REAL, "Volume" REAL
                );
                """ # 지표 컬럼은 저장하지 않음 (필요시 다시 계산)
                conn.execute(create_table_query)

                existing_df = pd.read_sql(f"SELECT \"Open time\" FROM {table_name}", conn, parse_dates=['Open time'])
                df_to_save = df_to_save[~df_to_save['Open time'].isin(existing_df['Open time'])]
            except pd.io.sql.DatabaseError: 
                logger.info(f"테이블 '{table_name}'이(가) 존재하지 않아 새로 생성합니다.")
            except sqlite3.OperationalError as e_op: # 테이블이 아직 없을 때 SELECT에서 발생 가능
                 if "no such table" in str(e_op).lower():
                    logger.info(f"테이블 '{table_name}'이(가) 존재하지 않아 새로 생성합니다. (OperationalError)")
                 else: raise # 다른 OperationalError는 다시 발생시킴
            
            if not df_to_save.empty:
                # 저장할 컬럼만 선택 (지표 제외)
                cols_to_store_in_db = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
                df_to_save[cols_to_store_in_db].to_sql(table_name, conn, if_exists='append', index=False)
                logger.info(f"{len(df_to_save)}개의 새로운 K-line 데이터가 '{db_path}'의 '{table_name}' 테이블에 저장되었습니다.")
            else:
                logger.info(f"새롭게 저장할 K-line 데이터가 없습니다 (모두 중복).")
    except sqlite3.Error as e_sql:
        logger.error(f"SQLite 오류 발생 (데이터 저장 중): {e_sql}")
    except Exception as e:
        logger.error(f"데이터 저장 중 알 수 없는 오류 발생: {e}", exc_info=True)


def main_trading_job():
    global last_signal_check_time, current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price
    kst = timezone(timedelta(hours=9), 'KST')
    logger.info(f"\n{'-'*20} 주기적 매매 로직 시작: {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')} {'-'*20}")
    try:
        if not initialize_exchange_info(): logger.error("거래소 정보 초기화 실패."); return
        lev_info = client.futures_leverage_bracket(symbol=TRADING_PARAMS['symbol'])
        act_lev = 0
        if lev_info and isinstance(lev_info, list) and len(lev_info) > 0:
            for item in lev_info:
                if item.get('symbol') == TRADING_PARAMS['symbol'] and item.get('brackets') and len(item['brackets']) > 0:
                    act_lev = int(item['brackets'][0]['initialLeverage']); break
        if act_lev == 0: logger.warning(f"{TRADING_PARAMS['symbol']} 레버리지 가져오기 실패. API 응답: {lev_info}")
        elif act_lev != TRADING_PARAMS['leverage_config']:
            logger.info(f"레버리지 변경: {act_lev} -> {TRADING_PARAMS['leverage_config']}"); client.futures_change_leverage(symbol=TRADING_PARAMS['symbol'], leverage=TRADING_PARAMS['leverage_config'])
    except Exception as e_setup: logger.error(f"초기 설정 중 오류: {e_setup}", exc_info=True); send_email_notification(f"봇 오류: 초기 설정 실패", f"오류 내용: {e_setup}"); return

    check_position_status_and_sync()
    if current_position_side != 'None':
        # --- 하이브리드 청산 프로토콜 ---
        current_market_price = df_proc.iloc[-1]['Close'] # Get latest close price from df_proc
        position_held_hours = (pd.Timestamp.now(tz='UTC') - last_signal_check_time).total_seconds() / 3600
        current_profit_usd = calculate_current_profit(current_market_price) # This returns actual USD profit

        profit_threshold_for_trail_value = entry_atr * TRADING_PARAMS['profit_threshold_for_trail']

        # 1. 시간 기반 청산 (Time Stop)
        # Check if position held longer than TIME_STOP_PERIOD_HOURS AND is currently unprofitable
        if position_held_hours > TRADING_PARAMS['time_stop_period_hours'] and current_profit_usd < 0:
            exit_position(reason="Time Stop (Unprofitable)")
            logger.info(f"포지션({current_position_side}) 시간 기반 청산. 신규 진입 확인 안 함.")
            logger.info(f"주기적 매매 로직 종료: {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'-'*60}\n")
            return

        # 2. ATR 트레일링 스탑 (activate only when profitable)
        if current_profit_usd >= profit_threshold_for_trail_value:
            # Call ATR trailing stop logic from src.exits.trailing.py
            # Need current_price, high, low, atr_value for update_and_check_atr_trailing_stop
            # df_proc has these values for the latest candle
            latest_candle = df_proc.iloc[-1]
            if update_and_check_atr_trailing_stop(
                position=None, # Placeholder, as current globals are used
                current_price=latest_candle['Close'],
                high=latest_candle['High'],
                low=latest_candle['Low'],
                atr_value=latest_candle[atr_col] # Use latest ATR
            ):
                exit_position(reason="ATR Trailing Stop Hit")
                logger.info(f"포지션({current_position_side}) ATR 트레일링 스탑 청산. 신규 진입 확인 안 함.")
                logger.info(f"주기적 매매 로직 종료: {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'-'*60}\n")
                return
        
        # If not exited by time stop or trailing stop, manage existing SL/TP
        manage_existing_position_sl_tp()

        logger.info(f"포지션({current_position_side}) 보유 중. 신규 진입 확인 안 함.")
        logger.info(f"주기적 매매 로직 종료: {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'-'*60}\n")
        return

    now_utc = pd.Timestamp.now(tz='UTC')
    interval_pd = TRADING_PARAMS['interval_primary'].replace('m','min').replace('h','H').replace('d','D').replace('w','W')
    current_candle_start_utc = now_utc.floor(interval_pd)
    target_candle_analyze_utc = current_candle_start_utc - pd.Timedelta(TRADING_PARAMS['interval_primary'])

    if last_signal_check_time is not None and last_signal_check_time >= target_candle_analyze_utc:
        logger.info(f"이미 최신 완성 캔들({target_candle_analyze_utc}, 로컬 마지막 확인: {last_signal_check_time}) 분석 완료. 대기.")
        logger.info(f"주기적 매매 로직 종료 (중복 방지): {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'-'*60}\n")
        return
        
    logger.info(f"새로운 완성 캔들({target_candle_analyze_utc}) 진입 신호 확인 시도...")
    df_raw = get_historical_data_primary(TRADING_PARAMS['symbol'], TRADING_PARAMS['interval_primary'], TRADING_PARAMS['data_fetch_limit_primary'])
    if df_raw is None or df_raw.empty: logger.warning(f"{TRADING_PARAMS['interval_primary']} 데이터 수집 실패."); return
    
    save_data_to_sqlite(df_raw, TRADING_PARAMS['db_path'], TRADING_PARAMS['klines_table_name'])

    df_proc = calculate_indicators(df_raw.copy())
    min_candles_sig = max(TRADING_PARAMS['ema_long_period'], TRADING_PARAMS['rsi_period'], TRADING_PARAMS['atr_period_sl']) + 5
    if df_proc is None or df_proc.empty or len(df_proc) < min_candles_sig: logger.warning(f"지표 계산 실패/데이터 부족 ({len(df_proc) if df_proc is not None else 0} < {min_candles_sig})."); return
    if df_proc.index[-1] < target_candle_analyze_utc: logger.warning(f"가져온 데이터 마지막 캔들({df_proc.index[-1]}) < 분석 대상({target_candle_analyze_utc}). 데이터 동기화 문제?"); return

    # --- Phase 1 Filters ---
    current_time_for_filters = df_proc.index[-1].to_pydatetime() # Use the timestamp of the last candle
    
    # 1. Circuit Breaker Check
    # MAX_CONSECUTIVE_LOSSES and COOLDOWN_PERIOD_BARS are imported from sizing.py
    if not can_open_new_trade(current_time_for_filters):
        logger.warning("서킷 브레이커 활성화: 신규 거래 진입 불가.")
        last_signal_check_time = target_candle_analyze_utc; save_position_state()
        return

    # 2. Temporal Filter Check
    if not is_trade_allowed_by_time(current_time_for_filters, TRADING_PARAMS['allowed_hours'], TRADING_PARAMS['blocked_days']):
        logger.warning(f"시간 필터 활성화: 현재 시간({current_time_for_filters.strftime('%H:%M')}) 또는 요일({current_time_for_filters.strftime('%A')})에 거래 불가.")
        last_signal_check_time = target_candle_analyze_utc; save_position_state()
        return

    # 3. Regime Filter Check
    # Need ADX and ATR values from df_proc
    # Assuming ADX is calculated and available in df_proc if use_adx_filter is True
    # Assuming ATR is available as atr_col
    adx_val = df_proc['ADX'][-1] if 'ADX' in df_proc.columns else None
    atr_val = df_proc[atr_col][-1] if atr_col in df_proc.columns else None
    current_close_price = df_proc['Close'][-1]

    if adx_val is None or atr_val is None or current_close_price <= 0:
        logger.warning("레짐 필터 체크를 위한 ADX/ATR/Close 값 부족. 필터 적용 불가.")
        # Decide whether to proceed or return. For safety, return.
        last_signal_check_time = target_candle_analyze_utc; save_position_state()
        return

    # Calculate atr_percent_value
    atr_percent_value = (atr_val / current_close_price) * 100

    if not is_favorable_regime(adx_val, atr_percent_value, TRADING_PARAMS['adx_threshold_regime'], TRADING_PARAMS['atr_percent_threshold_regime']):
        logger.warning(f"레짐 필터 활성화: 현재 레짐(ADX={adx_val:.2f}, ATR_Percent={atr_percent_value:.2f})은 거래에 불리함.")
        last_signal_check_time = target_candle_analyze_utc; save_position_state()
        return
    # --- End Phase 1 Filters ---

    signal, sl_calc, tp_calc = check_ema_crossover_signal(df_proc)
    last_signal_check_time = target_candle_analyze_utc; save_position_state()

    if signal != 0 and sl_calc is not None and tp_calc is not None:
        logger.info(f"진입 신호: Side={'Long' if signal==1 else 'Short'}, SL_Est={format_price(sl_calc)}, TP_Est={format_price(tp_calc)}")
        balance_usdt = get_account_balance('USDT')
        if balance_usdt < TRADING_PARAMS['min_trade_value_usdt']: logger.error(f"USDT 잔고({balance_usdt}) 부족."); return
        
        entry_est_size_calc = df_proc.iloc[-1]['Close']
        qty_order = calculate_position_size(entry_est_size_calc, sl_calc, balance_usdt)
        if qty_order <= 0: logger.warning("계산된 주문 수량 0 이하. 주문 안 함."); return

        logger.info(f"신규 {'Long' if signal == 1 else 'Short'} 주문 준비: Qty={qty_order}, Entry(ForSize)={entry_est_size_calc}")
        order_side = SIDE_BUY if signal == 1 else SIDE_SELL
        order_res = place_market_order(TRADING_PARAMS['symbol'], order_side, qty_order)

        if order_res and 'orderId' in order_res:
            order_id = order_res['orderId']; logger.info(f"시장가 주문 성공! ID:{order_id}. 체결 확인 중...")
            time.sleep(2); entry_p_fill, qty_fill = None, 0.0
            for attempt in range(5):
                try:
                    order_info = client.futures_get_order(symbol=TRADING_PARAMS['symbol'], orderId=order_id)
                    if order_info and order_info.get('status') == 'FILLED':
                        avg_p, exec_q = order_info.get('avgPrice','0'), order_info.get('executedQty','0')
                        if float(avg_p) > 0 and float(exec_q) > 0: entry_p_fill, qty_fill = float(avg_p), float(exec_q); logger.info(f"주문(ID:{order_id}) 체결: AvgP={entry_p_fill}, ExecQ={qty_fill}"); break
                        else: logger.warning(f"주문(ID:{order_id}) 체결, 가격/수량 비정상 (AvgP:{avg_p}, ExecQ:{exec_q}). 시도 {attempt+1}/5")
                    else: logger.info(f"주문(ID:{order_id}) 체결 대기 (Status:{order_info.get('status','N/A')}). 시도 {attempt+1}/5")
                except Exception as e_ord_chk: logger.error(f"API 오류 (주문 정보 조회 ID:{order_id}): {e_ord_chk}")
                if attempt < 4: time.sleep(attempt + 1)
            
            if not entry_p_fill or qty_fill < TRADING_PARAMS['min_contract_size']:
                logger.error(f"주문(ID:{order_id}) 체결 정보 확인 실패/수량 부족. 필요시 수동 처리!"); send_email_notification(f"봇 오류: 주문 체결 실패 (ID:{order_id})", f"주문 ID {order_id}의 체결 정보를 확인하지 못했거나 수량이 부족합니다. 수동 확인 및 조치가 필요합니다."); reset_position_state()
            else:
                atr_sl_tp = df_proc.iloc[-1].get(atr_col)
                if pd.isna(atr_sl_tp) or atr_sl_tp <= 0: logger.error(f"SL/TP 계산용 ATR 유효하지 않음({atr_sl_tp}). 비상 청산!"); place_market_order(TRADING_PARAMS['symbol'], SIDE_SELL if signal==1 else SIDE_BUY, qty_fill); reset_position_state(); return
                sl_dist_re = atr_sl_tp * TRADING_PARAMS['atr_multiplier_sl']
                actual_sl, actual_tp = (entry_p_fill - sl_dist_re, entry_p_fill + (sl_dist_re * TRADING_PARAMS['risk_reward_ratio'])) if signal == 1 else (entry_p_fill + sl_dist_re, entry_p_fill - (sl_dist_re * TRADING_PARAMS['risk_reward_ratio']))
                if (signal == 1 and (actual_sl >= entry_p_fill or actual_tp <= entry_p_fill or actual_sl >= actual_tp)) or \
                   (signal == -1 and (actual_sl <= entry_p_fill or actual_tp >= entry_p_fill or actual_sl <= actual_tp)) or \
                   pd.isna(actual_sl) or pd.isna(actual_tp):
                    logger.error(f"실체결가 기반 SL/TP 유효하지 않음. E={entry_p_fill}, SL={actual_sl}, TP={actual_tp}. 비상 청산!"); place_market_order(TRADING_PARAMS['symbol'], SIDE_SELL if signal==1 else SIDE_BUY, qty_fill); reset_position_state(); return
                current_position_side, current_position_quantity, entry_price, stop_loss_price, take_profit_price, entry_atr = ('Long' if signal==1 else 'Short'), qty_fill, entry_p_fill, actual_sl, actual_tp, atr_sl_tp
                logger.info(f"포지션 업데이트: Side={current_position_side}, Qty={current_position_quantity}, Entry={entry_price}, SL={format_price(stop_loss_price)}, TP={format_price(take_profit_price)}")
                sl_tp_side = SIDE_SELL if signal == 1 else SIDE_BUY
                sl_ok, tp_ok = place_stop_loss_order(TRADING_PARAMS['symbol'], sl_tp_side, current_position_quantity, stop_loss_price), place_take_profit_order(TRADING_PARAMS['symbol'], sl_tp_side, current_position_quantity, take_profit_price)
                if not sl_ok or not tp_ok: logger.error("SL/TP 주문 설정 실패! 비상 청산!"); place_market_order(TRADING_PARAMS['symbol'], sl_tp_side, current_position_quantity); reset_position_state()
                else: logger.info("SL/TP 주문 설정 완료."); save_position_state(); send_email_notification(f"{current_position_side} 포지션 진입 ({TRADING_PARAMS['symbol']})", f"심볼: {TRADING_PARAMS['symbol']}\n포지션: {current_position_side}\n수량: {current_position_quantity}\n진입가: {entry_price}\nSL: {format_price(stop_loss_price)}\nTP: {format_price(take_profit_price)}")
        else: logger.info("신규 진입 신호 없음 또는 SL/TP 유효하지 않음.")
    logger.info(f"주기적 매매 로직 실행 종료: {datetime.now(tz=kst).strftime('%Y-%m-%d %H:%M:%S %Z')}\n{'-'*60}\n")

# --- 스케줄러 설정 ---
schedule.every().hour.at("00:05").do(main_trading_job) # "MM:SS" 형식
logger.info(f"자동매매 스케줄러 시작 (매 시간 00분 05초 실행, {TRADING_PARAMS['interval_primary']} 봉 기준)...")

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 프로그램 시작 시 이메일 알림
    start_time_kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S %Z')
    email_subject = "자동매매 봇 시작"
    email_body = f"EMA 교차 실매매 봇이 시작되었습니다.\n시작 시간: {start_time_kst}\n사용 파라미터: {json.dumps(TRADING_PARAMS, indent=2)}"
    send_email_notification(email_subject, email_body)

    logger.info("=" * 50); logger.info(f"EMA 교차 실매매 봇 시작 (Phase 1)"); logger.info(f"파라미터: {json.dumps(TRADING_PARAMS, indent=4)}"); logger.info("=" * 50 + "\n")
    if not initialize_exchange_info(): logger.error("거래소 정보 초기화 실패. 종료."); send_email_notification("봇 긴급 오류: 거래소 정보 초기화 실패", "거래소 정보를 초기화하는데 실패하여 봇을 종료합니다."); exit()
    load_position_state()
    
    # TODO Phase 1: `supervisor`를 사용하여 이 스크립트를 백그라운드에서 안정적으로 실행하고,
    # 예기치 않은 종료 시 자동으로 재시작하도록 설정하는 것이 중요합니다.
    # supervisor 설정 예시 (supervisor.conf):
    # [program:my_trading_bot]
    # command=/Users/choiyunseong/BitAstro/venv/bin/python /Users/choiyunseong/BitAstro/real_M1.py
    # directory=/Users/choiyunseong/BitAstro
    # autostart=true
    # autorestart=true
    # stderr_logfile=/Users/choiyunseong/BitAstro/logs/my_trading_bot_err.log
    # stdout_logfile=/Users/choiyunseong/BitAstro/logs/my_trading_bot_out.log
    # user=choiyunseong (M2 Mac Mini 사용자 계정)

    try:
        main_trading_job() 
        while True: schedule.run_pending(); time.sleep(1)
    except KeyboardInterrupt: logger.info("Ctrl+C 감지. 종료 중...")
    except Exception as e: logger.error(f"메인 루프 오류: {e}", exc_info=True); send_email_notification("봇 중대 오류 발생!", f"메인 루프에서 다음 오류 발생:\n{e}\n\n로그 파일을 확인하세요.")
    finally:
        logger.warning("프로그램 종료 절차..."); send_email_notification("봇 프로그램 종료됨", "실매매 봇 프로그램이 종료되었습니다. 상태를 확인해주세요.")
        try:
            orders = client.futures_get_open_orders(symbol=TRADING_PARAMS['symbol'])
            for o in orders:
                if not (o['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and o.get('reduceOnly', False)):
                    logger.info(f"미체결 비 SL/TP 주문(ID:{o['orderId']}, Type:{o['type']}) 취소 시도..."); client.futures_cancel_order(symbol=TRADING_PARAMS['symbol'], orderId=o['orderId'])
        except Exception as e_final: logger.error(f"종료 시 주문 정리 중 오류: {e_final}")
        if current_position_side != 'None': logger.warning(f"종료 시 포지션({current_position_side}, Qty:{current_position_quantity}) 열려있음. SL/TP 유지 확인 필요.")
        save_position_state(); logger.info("프로그램 정상 종료."); logging.info("=" * 50 + " EMA 교차 자동매매 스케줄러 종료 " + "=" * 50)
