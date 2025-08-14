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
    
    Args:
        X (np.array): 피처 데이터
        y (np.array): 레이블 데이터
        sequence_length (int): 입력 시퀀스의 길이
        forecast_horizon (int): 몇 스텝 앞을 예측할 것인지
    
    Returns:
        Tuple[np.array, np.array]: 시퀀스 형태의 X와 y
    """
    X_sequences, y_sequences = [], []
    # 루프 범위를 예측 호라이즌만큼 줄여서 인덱스 에러 방지
    for i in range(len(X) - sequence_length - forecast_horizon + 1):
        X_sequences.append(X[i:i + sequence_length])
        # 시퀀스의 마지막 시점에서 forecast_horizon 만큼 뒤의 값을 타겟으로 지정
        y_sequences.append(y[i + sequence_length + forecast_horizon - 1])
    return np.array(X_sequences), np.array(y_sequences)

def preprocess_for_transformer(
    input_path: str = "btc_dollar_bars_labeled.parquet",
    output_dir: str = "processed_data_regime_v5",
    sequence_length: int = 60,
    forecast_horizon: int = 5, # ✨ 5개 바 이후의 국면을 예측하도록 설정
    test_size: float = 0.2,
    val_size: float = 0.25
):
    """
    레이블링된 데이터를 로드하여 Transformer 모델 예측 학습을 위한
    시퀀스 데이터(.npy)와 전처리기(scaler, encoder)를 생성합니다.
    """
    print(f"🚀 Transformer 예측 학습용 데이터 전처리 시작 (Forecast Horizon: {forecast_horizon})...")
    
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
    
    print(f"\n- 시퀀스 데이터 생성 중... (Seq Len: {sequence_length}, Horizon: {forecast_horizon})")
    X_train_seq, y_train_seq = create_sequences_for_forecasting(X_train_scaled, y_train, sequence_length, forecast_horizon)
    X_val_seq, y_val_seq = create_sequences_for_forecasting(X_val_scaled, y_val, sequence_length, forecast_horizon)
    X_test_seq, y_test_seq = create_sequences_for_forecasting(X_test_scaled, y_test, sequence_length, forecast_horizon)
    
    print("\n- 생성된 시퀀스 데이터 Shape:")
    print(f"  - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")
    print(f"  - X_val:   {X_val_seq.shape}, y_val:   {y_val_seq.shape}")
    print(f"  - X_test:  {X_test_seq.shape}, y_test:  {y_test_seq.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_seq)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_seq)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_seq)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val_seq)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_seq)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test_seq)
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_regime_v5.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder_regime_v5.pkl'))
    
    print(f"\n💾 전처리된 데이터와 도구들이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    preprocess_for_transformer()
