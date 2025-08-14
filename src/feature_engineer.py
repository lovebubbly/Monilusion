# feature_engineer_v5_future_label.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import pickle
import gc

# --- 설정 ---
INPUT_PARQUET_PATH = 'C:\Monilusion\data\dollar_bars_BTCUSDT_2021-2025.parquet'
OUTPUT_DIR = 'processed_data_regime_v5' # v5: 미래 국면 예측용 데이터
SEQUENCE_LENGTH = 128
# *** 핵심 변경점: 10개 바 미래의 국면을 예측하도록 설정 ***
PREDICTION_HORIZON = 10 
TEST_SET_RATIO = 0.15
VALIDATION_SET_RATIO = 0.15
DTYPE = 'float32'

def load_and_clean_data(path):
    """Parquet 파일을 로드하고 정리합니다."""
    print(f"'{path}'에서 달러 바 데이터를 로드합니다...")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None
    df = df.sort_values('open_time').set_index('open_time')
    
    duplicate_count = df.index.duplicated().sum()
    if duplicate_count > 0:
        agg_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'value': 'sum', 'buy_value': 'sum', 'sell_value': 'sum'
        }
        if 'close_time' in df.columns: agg_rules['close_time'] = 'last'
        df = df.groupby(df.index).agg(agg_rules)
    
    if 'close_time' not in df.columns: df['close_time'] = df.index
    print("✅ 데이터 정리 완료")
    return df

def add_features(df):
    """피처 엔지니어링"""
    print("🔧 피처 엔지니어링을 시작합니다...")
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
        df['bar_duration_seconds'] = (df['close_time'] - df.index).dt.total_seconds().clip(lower=0)
    else:
        df['bar_duration_seconds'] = 300
    df['ofi'] = (df['buy_value'] - df['sell_value']) / (df['buy_value'] + df['sell_value'] + 1e-10)
    df['ofi_ema_10'] = ta.ema(df['ofi'], length=10)
    df['ofi_ema_30'] = ta.ema(df['ofi'], length=30)
    df['ofi_volatility'] = df['ofi'].rolling(20).std()
    print("📊 기술적 지표 계산 중...")
    df.ta.ema(length=10, append=True, col_names=('EMA_10',))
    df.ta.ema(length=30, append=True, col_names=('EMA_30',))
    df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True, col_names=('ATR_14',))
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']
    df['volume_ema_10'] = ta.ema(df['volume'], length=10)
    df['volume_ratio'] = df['volume'] / (df['volume_ema_10'] + 1e-10)
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype(DTYPE)
    initial_length = len(df)
    df.dropna(inplace=True)
    print(f"✅ 피처 엔지니어링 완료! 📉 NaN 제거: {initial_length - len(df):,}개 행 제거")
    return df

def define_current_regime(df):
    """(1단계) OFI와 모멘텀을 기반으로 *현재* 시장 국면을 정의합니다."""
    print("🎯 [1/2] 현재 시장 국면 정의 중...")
    df['price_momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    ofi_mean, ofi_std = df['ofi_ema_10'].mean(), df['ofi_ema_10'].std()
    price_mean, price_std = df['price_momentum'].mean(), df['price_momentum'].std()
    ofi_upper, ofi_lower = ofi_mean + 0.5 * ofi_std, ofi_mean - 0.5 * ofi_std
    price_upper, price_lower = price_mean + 0.3 * price_std, price_mean - 0.3 * price_std
    conditions = [
        (df['ofi_ema_10'] > ofi_upper) | (df['price_momentum'] > price_upper),
        (df['ofi_ema_10'] < ofi_lower) | (df['price_momentum'] < price_lower),
    ]
    choices = [1, 2] # 상승, 하락
    df['regime'] = np.select(conditions, choices, default=0) # 횡보
    return df

def create_future_label(df, horizon):
    """(2단계) 미래의 국면을 예측하기 위한 정답(레이블)을 생성합니다."""
    print(f"🎯 [2/2] {horizon}개 바 미래의 국면을 정답(label)으로 생성 중...")
    df['label'] = df['regime'].shift(-horizon)
    initial_len = len(df)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"🔪 미래 라벨 생성 후 NaN 제거: {initial_len - len(df)}개 행 제거")
    print("📊 최종 예측 목표(label) 분포:")
    label_dist = df['label'].value_counts(normalize=True).sort_index()
    for label, percentage in label_dist.items():
        print(f"   {label} ({['횡보','상승','하락'][label]}): {percentage:.1%}")
    return df

def create_sequences(features, labels, seq_length, dtype=DTYPE):
    """시퀀스 데이터셋 생성"""
    X, y = [], []
    for i in tqdm(range(len(features) - seq_length), desc="시퀀스 생성"):
        X.append(features[i:(i + seq_length)])
        y.append(labels[i + seq_length -1])
    return np.array(X, dtype=dtype), np.array(y)

def main():
    print("🚀 Feature Engineering v5.0 (Future Label Prediction) 시작!")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean_data(INPUT_PARQUET_PATH)
    if df is None: return
    df_featured = add_features(df)
    if df_featured is None: return
    
    df_regime_defined = define_current_regime(df_featured)
    labeled_df = create_future_label(df_regime_defined, PREDICTION_HORIZON)
    
    # *** 데이터 유출 방지를 위해 라벨 생성에 사용된 컬럼은 피처에서 제외 ***
    exclude_cols = [col for col in labeled_df.columns if any(x in col for x in ['BBL', 'BBM', 'BBU', 'BBB', 'BBP', 'MACD', 'MACDs', 'MACDh'])]
    exclude_cols += ['open', 'high', 'low', 'close', 'volume', 'value', 'buy_value', 'sell_value', 'close_time', 'regime', 'price_momentum', 'label']
    feature_cols = [col for col in labeled_df.columns if col not in list(dict.fromkeys(exclude_cols))]
    
    features_df = labeled_df[feature_cols]
    labels_series = labeled_df['label']
    
    print(f"\n📊 최종 피처 개수: {len(feature_cols)}개 / 최종 데이터 개수: {len(features_df):,}개")

    n_samples = len(features_df)
    n_test = int(n_samples * TEST_SET_RATIO)
    n_validation = int(n_samples * VALIDATION_SET_RATIO)
    n_train = n_samples - n_test - n_validation
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(features_df.iloc[:n_train]).astype(DTYPE)
    validation_features_scaled = scaler.transform(features_df.iloc[n_train:n_train + n_validation]).astype(DTYPE)
    test_features_scaled = scaler.transform(features_df.iloc[n_train + n_validation:]).astype(DTYPE)
    
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler_regime_v5.pkl')
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    
    print("\n--- 🎯 [3/3] 순차적 데이터 생성 및 저장 시작 ---")
    
    print("\n[1/3] Train Set...")
    X_train, y_train = create_sequences(train_features_scaled, labels_series.iloc[:n_train].values, SEQUENCE_LENGTH)
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    del X_train, y_train, train_features_scaled; gc.collect()
    
    print("\n[2/3] Validation Set...")
    X_val, y_val = create_sequences(validation_features_scaled, labels_series.iloc[n_train:n_train+n_validation].values, SEQUENCE_LENGTH)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    del X_val, y_val, validation_features_scaled; gc.collect()

    print("\n[3/3] Test Set...")
    X_test, y_test = create_sequences(test_features_scaled, labels_series.iloc[n_train+n_validation:].values, SEQUENCE_LENGTH)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    del X_test, y_test, test_features_scaled; gc.collect()

    print("\n" + "=" * 60)
    print("🎉 Feature Engineering v5.0 (Future Label) 완료! 🎉")

if __name__ == '__main__':
    main()
