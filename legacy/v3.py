import pandas as pd
import numpy as np
from tqdm import tqdm

# tqdm이 pandas와 잘 작동하도록 설정
tqdm.pandas()

def label_market_regime_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    피처가 추가된 데이터프레임에 시장 국면(Regime) 레이블을 추가합니다. (V3: 횡보 조건 완화 및 균형 조정)
    - 'Ranging' 국면의 정의를 완화하여 더 많은 데이터를 포착합니다.
    - 와이코프 이론 기반의 'Accumulation'/'Distribution' 개념은 유지합니다.
    - 모호한 'Undefined'를 줄이기 위해 레이블링 우선순위를 조정합니다.

    Args:
        df (pd.DataFrame): feature_engineering_module.py로 생성된 피처들이 포함된 데이터프레임.

    Returns:
        pd.DataFrame: 'regime' 컬럼이 추가된 데이터프레임.
    """
    print("🚀 시장 국면(Regime) 레이블링을 시작합니다... (V3: 최종 버전)")

    # 안전한 작업을 위해 원본 데이터프레임 복사
    df_labeled = df.copy()

    # --- 국면 정의를 위한 조건 설정 ---
    
    # 1. 추세 방향성 (Trend Direction)
    is_bull_trend = (df_labeled['ema_20'] > df_labeled['ema_50']) & (df_labeled['ema_50'] > df_labeled['ema_100'])
    is_bear_trend = (df_labeled['ema_20'] < df_labeled['ema_50']) & (df_labeled['ema_50'] < df_labeled['ema_100'])

    # 2. 추세 강도 (Trend Strength)
    is_strong_trend = df_labeled['adx'] > 25
    
    # 3. 횡보성 (Choppiness) - V3 수정: Ranging 정의를 위한 조건 완화
    # 추세가 명확하지 않고(adx < 22), 시장이 방향성을 잃었을 때(chop > 60)를 횡보로 정의
    is_trendless = df_labeled['adx'] < 22
    is_choppy = df_labeled['chop'] > 60
    is_ranging = is_trendless & is_choppy

    # 4. 와이코프 이론 기반 횡보장 세분화
    is_above_long_term_ma = df_labeled['close'] > df_labeled['ema_200']
    
    # --- np.select를 사용한 효율적인 조건부 레이블링 (V3 수정) ---
    conditions = [
        is_bull_trend & is_strong_trend,
        is_bear_trend & is_strong_trend,
        is_ranging & is_above_long_term_ma,
        is_ranging & ~is_above_long_term_ma,
        is_bull_trend,
        is_bear_trend,
    ]

    choices = [
        'Strong_Bull_Trend',
        'Strong_Bear_Trend',
        'Ranging_Accumulation', 
        'Ranging_Distribution', 
        'Weak_Bull',
        'Weak_Bear',
    ]

    # 기본값 설정: 위 조건에 해당하지 않으면 약한 추세로 우선 분류
    default_choice = np.where(df_labeled['ema_50'] > df_labeled['ema_100'], 'Weak_Bull', 'Weak_Bear')
    
    df_labeled['regime_temp'] = np.select(conditions, choices, default=None)
    df_labeled['regime'] = df_labeled['regime_temp'].where(pd.notna(df_labeled['regime_temp']), default_choice)
    df_labeled.drop(columns=['regime_temp'], inplace=True)

    print("✅ 시장 국면 레이블링 완료!")
    
    return df_labeled

# --------------------------------------------------------------------------
# 🚀 이 모듈을 직접 실행하여 테스트하는 예시
# --------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        # ✨ 입력 파일 경로를 V2 피처 파일로 변경!
        input_parquet_path = "btc_dollar_bars_with_features_v2.parquet"
        featured_df = pd.read_parquet(input_parquet_path)
        print(f"'{input_parquet_path}' 파일 로드 성공. (총 {len(featured_df)}개 바)")
        
        # V3 레이블링 함수 실행
        labeled_df = label_market_regime_v3(featured_df)
        
        print("\n--- 국면별 데이터 분포 (V3 + V2 Features) ---")
        regime_distribution = labeled_df['regime'].value_counts(normalize=True) * 100
        print(regime_distribution.to_string(float_format="%.2f%%"))

        # ✨ 최종 결과물이니, 출력 파일 이름도 명확하게!
        output_parquet_path = "btc_dollar_bars_final_labeled.parquet"
        labeled_df.to_parquet(output_parquet_path, index=False)
        print(f"\n💾 최종 V3 레이블링 데이터가 '{output_parquet_path}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 에러: '{input_parquet_path}' 파일을 찾을 수 없습니다. 이전 단계에서 파일을 생성했는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")