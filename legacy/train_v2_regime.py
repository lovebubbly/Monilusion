import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
import multiprocessing as mp


# --- 모델 정의 (V2: [CLS] 토큰 적용 및 구조 개선) ---
class PositionalEncoding(nn.Module):
    """ 위치 정보를 인코딩하여 모델에 주입합니다. """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5001):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TradingTransformerV2(nn.Module):
    """
    [CLS] 토큰과 증설된 용량을 갖춘 V2 트레이딩 트랜스포머 모델
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_classes):
        super(TradingTransformerV2, self).__init__()
        self.d_model = d_model
        
        # ✨ V2 핵심: [CLS] 토큰을 학습 가능한 파라미터로 정의
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.encoder_input = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_input.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (batch_size, seq_len, num_features)
        src = self.encoder_input(src) * np.sqrt(self.d_model)
        
        # [CLS] 토큰을 모든 시퀀스의 맨 앞에 추가
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1) # Shape: (batch_size, seq_len + 1, d_model)
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # [CLS] 토큰의 최종 출력값만 사용하여 분류 (시퀀스 전체 정보 압축)
        cls_output = output[:, 0, :]
        
        output = self.decoder(cls_output)
        return output

# --- Focal Loss (클래스 불균형 데이터에 효과적) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            F_loss = alpha_t * F_loss
        return torch.mean(F_loss)

# --- 설정 및 유틸리티 함수 ---
DATA_DIR = 'processed_data_regime_final'
MODEL_SAVE_DIR = 'saved_models_regime_transformer_v2'

# ✨ V2 핵심: 모델 용량 증설 및 하이퍼파라미터 조정
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
    """ .npy 데이터 파일을 로드하고 Tensor로 변환합니다. """
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

def run_evaluation(model, data_loader, criterion, device, label_encoder):
    """ 검증 또는 테스트 데이터셋으로 모델 성능을 평가합니다. """
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
    f1 = report['macro avg']['f1-score']
    
    return avg_loss, f1, all_labels, all_preds

# --- 메인 학습 함수 ---
def main():
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

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [학습]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_pbar.set_postfix(loss=loss.item())

        val_loss, val_f1, _, _ = run_evaluation(model, val_loader, criterion, device, le)
        scheduler.step(val_f1)
        
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model_regime_v2.pth'))
            print(f"** 최고 검증 F1 Score 갱신! 모델 저장됨. (F1: {best_val_f1:.4f}) **")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n! {EARLY_STOPPING_PATIENCE} 에포크 동안 검증 F1 Score 개선 없어 조기 종료.")
            break
            
    # --- 최종 모델 성능 평가 (Test Set) ---
    print("\n--- 최종 모델 성능 평가 (Test Set) ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'best_model_regime_v2.pth')))
    
    X_test, y_test = load_data(DATA_DIR, is_test=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    _, _, all_test_labels, all_test_preds = run_evaluation(model, test_loader, criterion, device, le)
    
    print("\n" + classification_report(all_test_labels, all_test_preds, target_names=le.classes_, zero_division=0))
    
    cm = confusion_matrix(all_test_labels, all_test_preds, labels=range(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Final Test Set Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == '__main__':
    # Windows에서 multiprocessing DataLoader 사용 시 오류 방지를 위해
    if os.name == 'nt':
        mp.freeze_support()
    main()
