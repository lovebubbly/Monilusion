# feature_engineering.py

import pandas as pd
import pandas_ta as ta

def add_features(df):
    """
    주어진 데이터프레임에 기술적 분석 지표들을 추가한다.
    모든 피처 계산은 여기서 중앙 관리한다.
    """
    # 기존 OHLCV 데이터 복사
    df_out = df.copy()

    # Pandas TA 라이브러리를 사용하여 다양한 지표 추가
    # 예: 이동평균, RSI, MACD, Stoch, Bollinger Bands 등
    df_out.ta.ema(length=10, append=True)
    df_out.ta.ema(length=20, append=True)
    df_out.ta.ema(length=50, append=True)
    df_out.ta.ema(length=100, append=True)
    df_out.ta.ema(length=200, append=True)
    
    df_out.ta.rsi(length=14, append=True)
    df_out.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_out.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    
    # 볼린저 밴드 및 밴드폭(BBW) 계산 (레이블링에 사용)
    bbands = df_out.ta.bbands(length=20, std=2, append=True)
    df_out['BBW_20_2.0'] = bbands['BBB_20_2.0'] # Bollinger Band Width
    
    # ADX 계산 (레이블링에 사용)
    df_out.ta.adx(length=14, append=True)
    
    # ATR 계산 (리스크 관리에 활용 가능)
    df_out.ta.atr(length=14, append=True)

    # 결측치 처리 (초반 계산 안되는 부분)
    df_out.dropna(inplace=True)
    
    return df_out