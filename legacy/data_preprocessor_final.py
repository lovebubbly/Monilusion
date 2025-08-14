import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm import tqdm

def create_sequences_for_forecasting(X, y, sequence_length, forecast_horizon):
    """ 
    데이터셋을 예측(Forecasting)을 위한 시퀀스 형태로 변환합니다.
    """
    X_sequences, y_sequences = [], []
    for i in tqdm(range(len(X) - sequence_length - forecast_horizon + 1), desc="시퀀스 생성 중"):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length + forecast_horizon - 1])
    return np.array(X_sequences), np.array(y_sequences)

def preprocess_for_transformer_final(
    input_path: str = "btc_dollar_bars_final_labeled.parquet",
    output_dir: str = "processed_data_regime_final",
    sequence_length: int = 60,
    forecast_horizon: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.25
):
    """
    최종 레이블링된 데이터를 로드하여 Transformer 모델 학습을 위한
    시퀀스 데이터(.npy)와 전처리기(scaler, encoder)를 생성합니다.
    """
    print(f"🚀 최종 데이터셋 전처리 시작 (Forecast Horizon: {forecast_horizon})...")
    
    try:
        df = pd.read_parquet(input_path)
        print(f"✅ '{input_path}' 로드 성공. (총 {len(df)}개 바)")
    except FileNotFoundError:
        print(f"❌ 에러: '{input_path}' 파일을 찾을 수 없습니다.")
        return

    features_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'regime']
    X = df.drop(columns=features_to_drop, errors='ignore').values
    y = df['regime'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=test_size, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("\n- 피처 스케일링 완료 (StandardScaler)")
    
    X_train_seq, y_train_seq = create_sequences_for_forecasting(X_train_scaled, y_train, sequence_length, forecast_horizon)
    X_val_seq, y_val_seq = create_sequences_for_forecasting(X_val_scaled, y_val, sequence_length, forecast_horizon)
    X_test_seq, y_test_seq = create_sequences_for_forecasting(X_test_scaled, y_test, sequence_length, forecast_horizon)
    
    print("\n- 생성된 시퀀스 데이터 Shape:")
    print(f"  - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_seq)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_seq)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_seq)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val_seq)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_seq)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test_seq)
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_regime_final.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder_regime_final.pkl'))
    
    print(f"\n💾 최종 전처리 데이터와 도구들이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    preprocess_for_transformer_final()
