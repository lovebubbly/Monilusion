import pandas as pd
import numpy as np
from tqdm import tqdm

# tqdm이 pandas와 잘 작동하도록 설정
tqdm.pandas()

def label_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    피처가 추가된 데이터프레임에 시장 국면(Regime) 레이블을 추가합니다.
    규칙은 EMA의 방향성, ADX의 추세 강도, CHOP의 추세/횡보성을 종합적으로 판단합니다.
    (횡보 국면을 통합하여 클래스 불균형을 완화한 v2 버전)

    Args:
        df (pd.DataFrame): feature_engineering_module.py로 생성된 피처들이 포함된 데이터프레임.

    Returns:
        pd.DataFrame: 'regime' 컬럼이 추가된 데이터프레임.
    """
    print("🚀 시장 국면(Regime) 레이블링을 시작합니다... (v2: 횡보 국면 통합)")

    # 안전한 작업을 위해 원본 데이터프레임 복사
    df_labeled = df.copy()

    # --- 국면 정의를 위한 조건 설정 ---
    # 너의 연구 계획에 기반하여, 각 지표를 통해 시장 상태를 판단하는 기준을 정의
    
    # 1. 추세 방향성 (Trend Direction) - 더 명확한 추세를 위해 ema_100 조건 강화
    is_bull_trend = (df_labeled['ema_20'] > df_labeled['ema_50']) & (df_labeled['ema_50'] > df_labeled['ema_100'])
    is_bear_trend = (df_labeled['ema_20'] < df_labeled['ema_50']) & (df_labeled['ema_50'] < df_labeled['ema_100'])

    # 2. 추세 강도 (Trend Strength)
    is_strong_trend = df_labeled['adx'] > 25
    
    # 3. 횡보성 (Choppiness) - 이전보다 범위를 넓혀 횡보를 더 잘 포착
    is_choppy_market = df_labeled['chop'] > 50

    # --- np.select를 사용한 효율적인 조건부 레이블링 (v2 수정) ---
    # 우선순위: 강한 추세 -> 약한 추세 -> 횡보 순으로 판단
    conditions = [
        is_bull_trend & is_strong_trend,    # 1. 강세 추세 (Strong_Bull_Trend)
        is_bear_trend & is_strong_trend,    # 2. 약세 추세 (Strong_Bear_Trend)
        is_bull_trend,                      # 3. 약한 강세 (Weak_Bull) - 강한 추세가 아닌 나머지 상승 추세
        is_bear_trend,                      # 4. 약한 약세 (Weak_Bear) - 강한 추세가 아닌 나머지 하락 추세
        is_choppy_market                    # 5. 횡보 (Ranging) - Volatile과 Quiet을 통합
    ]

    # 각 조건에 해당하는 레이블 정의 (v2 수정)
    choices = [
        'Strong_Bull_Trend',
        'Strong_Bear_Trend',
        'Weak_Bull',
        'Weak_Bear',
        'Ranging'
    ]

    # np.select를 사용하여 'regime' 컬럼 생성, 나머지는 'Transition'
    df_labeled['regime'] = np.select(conditions, choices, default='Transition')

    print("✅ 시장 국면 레이블링 완료!")
    
    return df_labeled

# --------------------------------------------------------------------------
# 🚀 이 모듈을 직접 실행하여 테스트하는 예시
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # 이전에 생성한, 피처가 포함된 결과 파일을 로드
    try:
        input_parquet_path = "btc_dollar_bars_with_features.parquet"
        featured_df = pd.read_parquet(input_parquet_path)
        print(f"'{input_parquet_path}' 파일 로드 성공. (총 {len(featured_df)}개 바)")
        
        # 시장 국면 레이블링 함수 실행
        labeled_df = label_market_regime(featured_df)
        
        # 결과 확인
        print("\n--- 레이블링 결과 (상위 5개 행) ---")
        print(labeled_df[['timestamp', 'close', 'regime']].head())
        
        print("\n--- 국면별 데이터 분포 ---")
        regime_distribution = labeled_df['regime'].value_counts(normalize=True) * 100
        print(regime_distribution.to_string(float_format="%.2f%%"))

        # 결과 파일 저장 (선택 사항)
        output_parquet_path = "btc_dollar_bars_labeled.parquet"
        labeled_df.to_parquet(output_parquet_path, index=False)
        print(f"\n💾 레이블링된 데이터가 '{output_parquet_path}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 에러: '{input_parquet_path}' 파일을 찾을 수 없습니다. 이전 단계에서 파일을 생성했는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
