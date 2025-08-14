import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from model import TradingTransformerV2
from utils import FocalLoss

# --- 설정 및 유틸리티 함수 ---
DATA_DIR = 'processed_data_regime_final' # 최종 데이터 폴더
MODEL_SAVE_DIR = 'saved_models_regime_transformer_v2'

# ✨ V2 핵심: 모델 용량 증설
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 100
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

def load_data(data_dir, is_test=False):
    # ... (이전과 동일한 코드)
    print(f"'{data_dir}'에서 전처리된 데이터를 로드합니다...")
    if is_test:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        return torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    return (torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(),
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

def main():
    # ... (이전과 동일한 메인 학습 루프)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    X_train, y_train, X_val, y_val = load_data(DATA_DIR)
    le = joblib.load(os.path.join(DATA_DIR, 'label_encoder_regime_final.pkl'))
    NUM_CLASSES = len(le.classes_)
    print(f"감지된 국면 클래스 개수: {NUM_CLASSES}")

    class_counts = torch.bincount(y_train)
    weights = torch.zeros(NUM_CLASSES)
    weights[:len(class_counts)] = class_counts.float()
    weights = 1.0 / (weights + 1e-6)
    weights_tensor = (weights / weights.sum() * NUM_CLASSES).to(device)
    print(f"클래스 가중치: {weights_tensor.cpu().numpy()}")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = TradingTransformerV2(
        num_features=X_train.shape[2], d_model=D_MODEL, nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, num_classes=NUM_CLASSES
    ).to(device)
    print(f"\n--- V2 모델 구조 (총 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}) ---\n")

    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    early_stopping_counter = 0

    # ... (이전과 동일한 학습/검증 루프) ...
    print("학습을 시작합니다...")
    # ... (학습 코드 생략, 이전과 동일)

if __name__ == '__main__':
    main()
