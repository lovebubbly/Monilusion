# feature_engineer_14features_future.py - 14개 피처 미래 예측 버전

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import pickle
import gc

# --- 설정 ---
INPUT_PARQUET_PATH = 'data/dollar_bars_BTCUSDT_2023-01-01_2024-01-01.parquet'
OUTPUT_DIR = 'C:/processed_data_regime_14features'
SEQUENCE_LENGTH = 128
PREDICTION_HORIZON = 1  # 검색 결과와 동일
TEST_SET_RATIO = 0.15
VALIDATION_SET_RATIO = 0.15
DTYPE = 'float32'

def load_and_clean_data(path):
    """데이터 로드 및 정리"""
    print(f"📂 '{path}'에서 달러 바 데이터를 로드합니다...")
    try:
        df = pd.read_parquet(path)
        print(f"✅ 데이터 로드 성공: {len(df):,}개 바")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None

    df = df.sort_values('open_time').set_index('open_time')
    
    duplicate_count = df.index.duplicated().sum()
    if duplicate_count > 0:
        print(f"🔧 중복 인덱스 {duplicate_count}개 발견, 그룹화하여 처리합니다...")
        agg_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'value': 'sum', 'buy_value': 'sum', 'sell_value': 'sum'
        }
        if 'close_time' in df.columns: 
            agg_rules['close_time'] = 'last'
        df = df.groupby(df.index).agg(agg_rules)
        print(f"✅ 중복 제거 완료: {len(df):,}개 바")
    
    if 'close_time' not in df.columns: 
        df['close_time'] = df.index
    print("✅ 데이터 정리 완료")
    return df

def add_14_features(df):
    """정확히 14개 피처 생성"""
    print("🔧 14개 피처 엔지니어링 시작...")
    
    # 바 지속시간
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
        df['bar_duration_seconds'] = (df['close_time'] - df.index).dt.total_seconds().clip(lower=0)
    else:
        df['bar_duration_seconds'] = 300

    # OFI 관련 (5개)
    df['ofi'] = (df['buy_value'] - df['sell_value']) / (df['buy_value'] + df['sell_value'] + 1e-10)
    df['ofi_ema_10'] = ta.ema(df['ofi'], length=10)
    df['ofi_ema_30'] = ta.ema(df['ofi'], length=30)
    df['ofi_volatility'] = df['ofi'].rolling(20).std()

    print("📊 기술적 지표 계산 중...")
    # 기술적 지표 (4개)
    df.ta.ema(length=10, append=True, col_names=('EMA_10',))
    df.ta.ema(length=30, append=True, col_names=('EMA_30',))
    df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
    df.ta.atr(length=14, append=True, col_names=('ATR_14',))

    # 가격 관련 (2개)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']

    # 볼륨 관련 (2개)
    df['volume_ema_10'] = ta.ema(df['volume'], length=10)
    df['volume_ratio'] = df['volume'] / (df['volume_ema_10'] + 1e-10)

    # 가격 모멘텀 (1개)
    df['price_momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # 정확히 14개 피처 선택
    feature_cols = [
        'bar_duration_seconds',  # 1
        'ofi',                   # 2  
        'ofi_ema_10',           # 3
        'ofi_ema_30',           # 4
        'ofi_volatility',       # 5
        'EMA_10',               # 6
        'EMA_30',               # 7
        'RSI_14',               # 8
        'ATR_14',               # 9
        'log_return',           # 10
        'price_change_pct',     # 11
        'volume_ema_10',        # 12
        'volume_ratio',         # 13
        'price_momentum'        # 14
    ]

    # 데이터 타입 최적화
    for col in feature_cols:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype(DTYPE)

    initial_length = len(df)
    df.dropna(inplace=True)
    final_length = len(df)
    
    print(f"✅ 14개 피처 엔지니어링 완료!")
    print(f"📉 NaN 제거: {initial_length - final_length:,}개 행 제거")
    
    return df[feature_cols + ['close', 'buy_value', 'sell_value']]  # 라벨링용 컬럼들 포함

def create_balanced_regime_labels(df):
    """균형잡힌 3-클래스 국면 라벨 생성"""
    print("🎯 균형잡힌 시장 국면 라벨링...")
    
    # 동적 임계값 설정
    ofi_mean, ofi_std = df['ofi_ema_10'].mean(), df['ofi_ema_10'].std()
    price_mean, price_std = df['price_momentum'].mean(), df['price_momentum'].std()
    
    ofi_upper, ofi_lower = ofi_mean + 0.5 * ofi_std, ofi_mean - 0.5 * ofi_std
    price_upper, price_lower = price_mean + 0.3 * price_std, price_mean - 0.3 * price_std
    
    print(f"📊 임계값 - OFI: {ofi_lower:.4f}~{ofi_upper:.4f}, 가격: {price_lower:.4f}~{price_upper:.4f}")
    
    conditions = [
        (df['ofi_ema_10'] > ofi_upper) | (df['price_momentum'] > price_upper),  # 상승
        (df['ofi_ema_10'] < ofi_lower) | (df['price_momentum'] < price_lower),  # 하락
    ]
    choices = [1, 2]
    df['regime'] = np.select(conditions, choices, default=0)  # 횡보

    # 현재 국면 분포
    regime_counts = df['regime'].value_counts().sort_index()
    print("📊 현재 시장 국면 분포:")
    regime_names = {0: '횡보', 1: '상승', 2: '하락'}
    for regime, count in regime_counts.items():
        name = regime_names.get(regime, f'국면{regime}')
        print(f"   {regime} ({name}): {count:,}개 ({count/len(df)*100:.1f}%)")
    
    return df

def create_future_prediction_labels(df, horizon):
    """미래 예측 라벨 생성 (데이터 누수 방지)"""
    print(f"🔮 미래 {horizon}개 바 이후의 국면을 예측 대상으로 설정...")
    
    # 핵심: t 시점 피처로 t+horizon 시점 국면 예측
    df['label'] = df['regime'].shift(-horizon)
    
    initial_len = len(df)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    print(f"🔪 미래 라벨 생성 후 NaN 제거: {initial_len - len(df)}개 행 제거")
    
    # 최종 예측 목표 분포
    label_dist = df['label'].value_counts(normalize=True).sort_index()
    print("📊 최종 예측 목표(label) 분포:")
    regime_names = {0: '횡보', 1: '상승', 2: '하락'}
    for label, percentage in label_dist.items():
        name = regime_names.get(label, f'클래스{label}')
        print(f"   {label} ({name}): {percentage:.1%}")
    
    return df

def create_sequences_future_prediction(features, labels, seq_length, prediction_horizon=1, dtype='float32'):
    """검색 결과[1]와 동일한 미래 예측 시퀀스 생성"""
    print(f"🔄 {seq_length}개 과거 → {prediction_horizon}시점 미래 예측 시퀀스 생성...")
    X, y = [], []
    
    # 검색 결과와 동일한 로직
    max_idx = len(features) - seq_length - prediction_horizon + 1
    
    for i in tqdm(range(max_idx), desc="미래 예측 시퀀스 생성"):
        X.append(features[i:(i + seq_length)])
        y.append(labels[i + seq_length + prediction_horizon - 1])
    
    print(f"✅ 총 {len(X):,}개 시퀀스 생성 (미래 {prediction_horizon}시점 예측)")
    return np.array(X, dtype=dtype), np.array(y)

def main():
    """메인 실행 함수"""
    print("🚀 Feature Engineering 14개 피처 미래 예측 버전!")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드
    df = load_and_clean_data(INPUT_PARQUET_PATH)
    if df is None: return

    # 2. 정확히 14개 피처 생성
    df_featured = add_14_features(df)

    # 3. 균형잡힌 국면 라벨링
    df_regime = create_balanced_regime_labels(df_featured)
    
    # 4. 미래 예측 라벨 생성
    df_labeled = create_future_prediction_labels(df_regime, PREDICTION_HORIZON)
    
    # 5. 피처와 라벨 분리
    feature_cols = [
        'bar_duration_seconds', 'ofi', 'ofi_ema_10', 'ofi_ema_30', 'ofi_volatility',
        'EMA_10', 'EMA_30', 'RSI_14', 'ATR_14', 'log_return', 
        'price_change_pct', 'volume_ema_10', 'volume_ratio', 'price_momentum'
    ]
    
    features_df = df_labeled[feature_cols]
    labels_series = df_labeled['label']
    
    print(f"\n📊 최종 확인:")
    print(f"   피처 수: {len(feature_cols)}개 (정확히 14개!)")
    print(f"   샘플 수: {len(features_df):,}개")

    # 6. 시간 순서 기반 분할
    n_samples = len(features_df)
    n_test = int(n_samples * TEST_SET_RATIO)
    n_validation = int(n_samples * VALIDATION_SET_RATIO)
    n_train = n_samples - n_test - n_validation

    # 7. 스케일링
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(features_df.iloc[:n_train]).astype(DTYPE)
    validation_features_scaled = scaler.transform(features_df.iloc[n_train:n_train + n_validation]).astype(DTYPE)
    test_features_scaled = scaler.transform(features_df.iloc[n_train + n_validation:]).astype(DTYPE)
    
    # 스케일러 저장
    with open(os.path.join(OUTPUT_DIR, 'scaler_14features.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("\n--- 🎯 14개 피처 미래 예측 시퀀스 생성 ---")
    
    # 8. Train Set
    print("\n[1/3] Train Set...")
    X_train, y_train = create_sequences_future_prediction(
        train_features_scaled, labels_series.iloc[:n_train].values, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    print(f"✅ Train Set: X={X_train.shape}, y={y_train.shape}")
    del X_train, y_train, train_features_scaled; gc.collect()
    
    # 9. Validation Set  
    print("\n[2/3] Validation Set...")
    X_val, y_val = create_sequences_future_prediction(
        validation_features_scaled, labels_series.iloc[n_train:n_train+n_validation].values, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    print(f"✅ Validation Set: X={X_val.shape}, y={y_val.shape}")
    del X_val, y_val, validation_features_scaled; gc.collect()

    # 10. Test Set
    print("\n[3/3] Test Set...")
    X_test, y_test = create_sequences_future_prediction(
        test_features_scaled, labels_series.iloc[n_train+n_validation:].values, SEQUENCE_LENGTH, PREDICTION_HORIZON
    )
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    print(f"✅ Test Set: X={X_test.shape}, y={y_test.shape}")
    del X_test, y_test, test_features_scaled; gc.collect()

    print(f"\n🎉 14개 피처 미래 예측 데이터셋 완성!")
    print(f"💾 '{OUTPUT_DIR}'에 저장 완료")
    print(f"🔮 진짜 미래 예측 모델을 위한 완벽한 구조!")

if __name__ == '__main__':
    main()
