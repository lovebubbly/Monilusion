import pandas as pd
import numpy as np
import pandas_ta as ta # TA-Lib 대신 사용이 간편한 pandas_ta 라이브러리 활용
from tqdm import tqdm

# tqdm이 pandas와 잘 작동하도록 설정
tqdm.pandas()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    달러 바 데이터프레임에 다양한 기술적 지표(피처)를 추가합니다.
    이 함수는 모델 학습과 실시간 추론 모두에서 사용될 공유 모듈의 핵심입니다.

    Args:
        df (pd.DataFrame): 'timestamp', 'open', 'high', 'low', 'close' 컬럼을 포함하는 달러 바 데이터.

    Returns:
        pd.DataFrame: 다양한 피처가 추가된 데이터프레임.
    """
    print("🚀 핵심 피처 엔지니어링을 시작합니다...")

    # 안전한 작업을 위해 원본 데이터프레임 복사
    df_featured = df.copy()

    # --- 1. 다중 시간대(Multi-Timeframe) 이동평균 피처 ---
    # 단기, 중기, 장기 추세를 파악하기 위함
    ema_short_periods = [10, 20]
    ema_mid_periods = [50, 100]
    ema_long_periods = [200]
    
    print("  - 이동평균(EMA) 피처 계산 중...")
    for period in tqdm(ema_short_periods + ema_mid_periods + ema_long_periods, desc="EMA 계산"):
        df_featured[f'ema_{period}'] = ta.ema(df_featured['close'], length=period)

    # --- 2. 모멘텀 및 추세 강도 피처 ---
    # 시장의 과매수/과매도 상태와 추세의 힘을 측정
    print("  - 모멘텀(RSI, MACD) 및 추세 강도(ADX, CHOP) 피처 계산 중...")
    
    # RSI (Relative Strength Index)
    df_featured['rsi'] = ta.rsi(df_featured['close'], length=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df_featured['close'], fast=12, slow=26, signal=9)
    df_featured['macd'] = macd['MACD_12_26_9']
    df_featured['macd_signal'] = macd['MACDs_12_26_9']
    df_featured['macd_hist'] = macd['MACDh_12_26_9']
    
    # ADX (Average Directional Index) - 추세의 강도를 나타냄 (방향성 X)
    adx = ta.adx(df_featured['high'], df_featured['low'], df_featured['close'], length=14)
    df_featured['adx'] = adx['ADX_14']

    # Choppiness Index (CHOP) - 시장의 '추세성' vs '횡보성' 측정
    # pandas_ta에는 CHOP가 직접 내장되어 있지 않으므로, 아래와 같이 수동으로 계산
    # (너의 연구에서 CHOP의 중요성을 강조했으므로 추가)
    atr_val = ta.atr(df_featured['high'], df_featured['low'], df_featured['close'], length=1)
    highest_high = df_featured['high'].rolling(window=14).max()
    lowest_low = df_featured['low'].rolling(window=14).min()
    chop_numerator = atr_val.rolling(window=14).sum()
    chop_denominator = highest_high - lowest_low
    df_featured['chop'] = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14)
    
    # --- 3. 변동성(Volatility) 피처 ---
    # 시장의 위험 수준과 가격 변동폭을 측정
    print("  - 변동성(ATR, Bollinger Bands) 피처 계산 중...")
    
    # ATR (Average True Range)
    df_featured['atr'] = ta.atr(df_featured['high'], df_featured['low'], df_featured['close'], length=14)
    
    # 볼린저 밴드 (Bollinger Bands)
    bollinger = ta.bbands(df_featured['close'], length=20, std=2)
    df_featured['bb_upper'] = bollinger['BBU_20_2.0']
    df_featured['bb_middle'] = bollinger['BBM_20_2.0']
    df_featured['bb_lower'] = bollinger['BBL_20_2.0']
    df_featured['bb_width'] = (df_featured['bb_upper'] - df_featured['bb_lower']) / df_featured['bb_middle']

    # --- 4. 거래량 기반 피처 (Volume Features) ---
    # 시장 참여자들의 관심도와 에너지 측정 (달러 바에는 'volume'이 없으므로, 향후 추가 가정)
    if 'volume' in df_featured.columns:
        print("  - 거래량(Volume) 피처 계산 중...")
        # 거래량 이동평균
        df_featured['volume_ema_20'] = ta.ema(df_featured['volume'], length=20)
        # 거래량 비율 (현재 거래량이 최근 평균 대비 얼마나 되는가)
        df_featured['volume_ratio'] = df_featured['volume'] / df_featured['volume_ema_20']
    else:
        print("⚠️ 'volume' 컬럼이 없어 거래량 기반 피처는 생성되지 않았습니다.")

    # --- 5. 최종 정리 ---
    # 계산 과정에서 발생한 결측치(NaN) 처리
    df_featured.dropna(inplace=True)
    df_featured.reset_index(drop=True, inplace=True)
    
    print("✅ 피처 엔지니어링 완료!")
    
    return df_featured

# --------------------------------------------------------------------------
# 🚀 이 모듈을 직접 실행하여 테스트하는 예시
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # 이전에 생성한 달러 바 결과 파일을 로드
    try:
        input_parquet_path = "btc_dollar_bars_optimized.parquet"
        dollar_bars_df = pd.read_parquet(input_parquet_path)
        print(f"'{input_parquet_path}' 파일 로드 성공. (총 {len(dollar_bars_df)}개 바)")
        
        # 피처 엔지니어링 함수 실행
        featured_df = engineer_features(dollar_bars_df)
        
        # 결과 확인
        print("\n--- 피처 생성 결과 (상위 5개 행) ---")
        print(featured_df.head())
        print("\n--- 생성된 피처 목록 ---")
        print(featured_df.columns.tolist())
        print(f"\n결측치 제거 후 최종 데이터 수: {len(featured_df)}개")

        # 결과 파일 저장 (선택 사항)
        output_parquet_path = "btc_dollar_bars_with_features.parquet"
        featured_df.to_parquet(output_parquet_path, index=False)
        print(f"\n💾 피처가 추가된 데이터가 '{output_parquet_path}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 에러: '{input_parquet_path}' 파일을 찾을 수 없습니다. 이전 단계에서 파일을 생성했는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

