import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm

# tqdm이 pandas와 잘 작동하도록 설정
tqdm.pandas()

def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    달러 바 데이터프레임에 다양한 기술적 지표(피처)를 추가합니다.
    (V2: 횡보장 탐지 강화를 위한 피처 추가)

    Args:
        df (pd.DataFrame): 'timestamp', 'open', 'high', 'low', 'close' 컬럼을 포함하는 달러 바 데이터.

    Returns:
        pd.DataFrame: 다양한 피처가 추가된 데이터프레임.
    """
    print("🚀 핵심 피처 엔지니어링을 시작합니다... (V2: 횡보 탐지 강화)")

    # 안전한 작업을 위해 원본 데이터프레임 복사
    df_featured = df.copy()

    # --- 1. 다중 시간대(Multi-Timeframe) 이동평균 피처 ---
    ema_periods = [10, 20, 50, 100, 200]
    print("  - 이동평균(EMA) 피처 계산 중...")
    for period in tqdm(ema_periods, desc="EMA 계산"):
        df_featured[f'ema_{period}'] = ta.ema(df_featured['close'], length=period)

    # --- 2. 모멘텀 및 추세 강도 피처 ---
    print("  - 모멘텀(RSI, MACD) 및 추세 강도(ADX, CHOP) 피처 계산 중...")
    df_featured['rsi'] = ta.rsi(df_featured['close'], length=14)
    macd = ta.macd(df_featured['close'], fast=12, slow=26, signal=9)
    df_featured['macd'] = macd['MACD_12_26_9']
    df_featured['macd_signal'] = macd['MACDs_12_26_9']
    df_featured['macd_hist'] = macd['MACDh_12_26_9']
    adx = ta.adx(df_featured['high'], df_featured['low'], df_featured['close'], length=14)
    df_featured['adx'] = adx['ADX_14']
    atr_val = ta.atr(df_featured['high'], df_featured['low'], df_featured['close'], length=1)
    highest_high = df_featured['high'].rolling(window=14).max()
    lowest_low = df_featured['low'].rolling(window=14).min()
    chop_numerator = atr_val.rolling(window=14).sum()
    chop_denominator = highest_high - lowest_low
    df_featured['chop'] = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14)
    
    # --- 3. 변동성(Volatility) 피처 ---
    print("  - 변동성(ATR, Bollinger Bands) 피처 계산 중...")
    df_featured['atr'] = ta.atr(df_featured['high'], df_featured['low'], df_featured['close'], length=14)
    bollinger = ta.bbands(df_featured['close'], length=20, std=2)
    df_featured['bb_upper'] = bollinger['BBU_20_2.0']
    df_featured['bb_middle'] = bollinger['BBM_20_2.0']
    df_featured['bb_lower'] = bollinger['BBL_20_2.0']
    df_featured['bb_width'] = (df_featured['bb_upper'] - df_featured['bb_lower']) / df_featured['bb_middle']

    # --- ✨ 4. V2 신규 피처: 횡보장 탐지 강화 ---
    print("  - V2 신규 피처 (횡보 탐지용) 계산 중...")
    short_window = 20 # 단기 기간 정의

    # 신규 피처 1: 가격 변동성 (Price Volatility)
    # 횡보장에서는 가격 변동성이 낮아지는 경향이 있음
    df_featured['price_volatility'] = df_featured['close'].rolling(window=short_window).std()

    # 신규 피처 2: 이동평균 교차 빈도 (MA Crossing Count)
    # 횡보장에서는 가격이 단기 이평선을 자주 위아래로 교차함
    ema_short = df_featured['ema_20'] # 20일 이평선을 기준으로
    price_above_ma = (df_featured['close'] > ema_short).astype(int)
    # 이전 시점과 현재 시점의 위치가 다른 경우를 교차로 간주
    crossings = (price_above_ma.diff() != 0).astype(int)
    df_featured['ma_cross_count'] = crossings.rolling(window=short_window).sum()

    # --- 5. 최종 정리 ---
    df_featured.dropna(inplace=True)
    df_featured.reset_index(drop=True, inplace=True)
    
    print("✅ 피처 엔지니어링 완료! (V2)")
    
    return df_featured

# --------------------------------------------------------------------------
# 🚀 이 모듈을 직접 실행하여 테스트하는 예시
# --------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        input_parquet_path = "btc_dollar_bars_optimized.parquet" # 원본 달러 바 파일
        dollar_bars_df = pd.read_parquet(input_parquet_path)
        print(f"'{input_parquet_path}' 파일 로드 성공. (총 {len(dollar_bars_df)}개 바)")
        
        # V2 피처 엔지니어링 함수 실행
        featured_df_v2 = engineer_features_v2(dollar_bars_df)
        
        print("\n--- V2 피처 생성 결과 (상위 5개 행) ---")
        # 새로 추가된 피처들을 확인하기 위해 마지막 컬럼들 출력
        print(featured_df_v2.tail())
        
        print("\n--- 생성된 피처 목록 ---")
        print(featured_df_v2.columns.tolist())
        print(f"\n결측치 제거 후 최종 데이터 수: {len(featured_df_v2)}개")

        # V2 결과 파일 저장
        output_parquet_path = "btc_dollar_bars_with_features_v2.parquet"
        featured_df_v2.to_parquet(output_parquet_path, index=False)
        print(f"\n💾 V2 피처가 추가된 데이터가 '{output_parquet_path}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 에러: '{input_parquet_path}' 파일을 찾을 수 없습니다. 원본 달러 바 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
