import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def train_regime_classifier(df: pd.DataFrame, test_size: float = 0.2):
    """
    레이블링된 데이터로 XGBoost 시장 국면 분류 모델을 학습하고 평가합니다.
    학습된 모델과 전처리 도구들을 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 'regime' 컬럼과 피처들이 포함된 데이터프레임.
        test_size (float): 테스트 데이터셋의 비율.
    """
    print("🚀 XGBoost 시장 국면 분류 모델 학습을 시작합니다...")

    # --- 1. 데이터 준비: 피처(X)와 타겟(y) 분리 ---
    # 타겟 변수 'regime'을 제외한 모든 컬럼을 피처로 사용
    # timestamp, open, high, low, close 등은 모델 학습에 직접 사용하지 않음
    features_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'regime']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['regime']

    print(f"✅ 사용된 피처 개수: {len(X.columns)}개")
    print(f"피처 목록: {X.columns.tolist()}")

    # --- 2. 타겟 레이블 인코딩 ---
    # 문자열 레이블(e.g., 'Strong_Bull_Trend')을 숫자(0, 1, 2...)로 변환
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("\n--- 레이블 인코딩 맵 ---")
    for i, class_name in enumerate(le.classes_):
        print(f"{i}: {class_name}")

    # --- 3. 데이터 분할: 학습용 vs 테스트용 ---
    # 시계열 데이터이므로, 순서를 유지하기 위해 shuffle=False 옵션 사용
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, shuffle=False
    )
    print(f"\n- 학습 데이터: {len(X_train)}개 / 테스트 데이터: {len(X_test)}개")

    # --- 4. 피처 스케일링 ---
    # 피처들의 단위를 통일시켜 모델 성능을 향상시킴
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("- 피처 스케일링 완료 (StandardScaler)")

    # --- 5. 클래스 불균형 처리를 위한 가중치 계산 ---
    # 데이터가 적은 클래스에 더 높은 가중치를 부여하여 학습을 유도
    # sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    # print("- 클래스 불균형 보정을 위한 샘플 가중치 계산 완료")
    # XGBoost는 scale_pos_weight 파라미터를 제공하므로, 여기서는 직접 가중치를 계산하는 대신
    # DMatrix 생성 시 `weight` 파라미터 사용을 고려할 수 있습니다.
    # 더 간단한 접근은 각 클래스의 비율을 계산하여 `scale_pos_weight`에 활용하는 것이지만
    # 다중 클래스에서는 직접적인 `scale_pos_weight` 설정이 복잡하므로,
    # 여기서는 XGBoost의 objective='multi:softmax' 기본 기능에 의존합니다.
    # 성능이 부족할 경우, sample_weight를 DMatrix에 전달하는 방식을 추후에 구현합니다.


    # --- 6. XGBoost 모델 학습 ---
    print("\n🔥 XGBoost 모델 학습 중... (잠시만 기다려주세요)")
    model = xgb.XGBClassifier(
        objective='multi:softmax', # 다중 클래스 분류
        num_class=len(le.classes_),
        n_estimators=200,          # 트리의 개수
        max_depth=5,               # 트리의 최대 깊이
        learning_rate=0.1,         # 학습률
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    print("✅ 모델 학습 완료!")

    # --- 7. 모델 성능 평가 ---
    print("\n--- 모델 성능 평가 (테스트 데이터) ---")
    y_pred = model.predict(X_test_scaled)
    
    # Classification Report 출력
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # --- 8. "지능 패키지" 저장 ---
    # 실매매 시스템에서 사용할 수 있도록 모델과 전처리 도구를 저장
    output_dir = "intelligence_package"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "regime_classifier.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    
    print(f"\n💾 '지능 패키지'가 '{output_dir}' 폴더에 저장되었습니다.")
    print(f"  - 모델: {model_path}")
    print(f"  - 스케일러: {scaler_path}")
    print(f"  - 인코더: {encoder_path}")


if __name__ == '__main__':
    try:
        input_parquet_path = "btc_dollar_bars_labeled.parquet"
        labeled_df = pd.read_parquet(input_parquet_path)
        print(f"'{input_parquet_path}' 파일 로드 성공. (총 {len(labeled_df)}개 바)")
        
        # 모델 학습 및 평가 함수 실행
        train_regime_classifier(labeled_df)

    except FileNotFoundError:
        print(f"❌ 에러: '{input_parquet_path}' 파일을 찾을 수 없습니다. 이전 단계에서 파일을 생성했는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
