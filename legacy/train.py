# train_v2_regime.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd


# model.py가 같은 디렉토리에 있다고 가정
# from model import TradingTransformer 

# --- 임시 TradingTransformer 모델 정의 (model.py가 없을 경우 대비) ---
# 실제 사용 시에는 model.py에서 임포트하는 것을 권장합니다.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TradingTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_classes):
        super(TradingTransformer, self).__init__()
        self.d_model = d_model
        self.encoder_input = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_input.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder_input(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # (batch, seq, feature) -> (batch, feature)
        # 각 시퀀스의 마지막 타임스텝의 출력을 사용
        output = output[:, -1, :] 
        output = self.decoder(output)
        return output
# --- 여기까지 임시 모델 정의 ---


# --- 하이퍼파라미터 및 설정 ---
DATA_DIR = 'processed_data_regime_v5'
MODEL_SAVE_DIR = 'saved_models_regime'
BATCH_SIZE = 64
LEARNING_RATE = 5e-5 # 0.00005
NUM_EPOCHS = 200
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 8
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15 # F1 Score 기준

# --- Focal Loss (불균형 데이터에 매우 효과적) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # alpha를 클래스 가중치로 사용하기 위해 수정
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss

        # alpha (클래스 가중치) 적용
        if self.alpha is not None:
            # alpha 텐서를 현재 target에 맞게 인덱싱
            alpha_t = self.alpha[targets]
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss


def load_data(data_dir):
    """지정된 디렉토리에서 .npy 파일들을 로드합니다."""
    print(f"'{data_dir}'에서 전처리된 데이터를 로드합니다...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    
    print("데이터 로드 완료.")
    return X_train, y_train, X_val, y_val

def calculate_metrics(y_true, y_pred, num_classes):
    """F1 Score, Precision, Recall 및 Confusion Matrix를 계산합니다."""
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return f1, precision, recall, cm

def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    X_train, y_train, X_val, y_val = load_data(DATA_DIR)
    
    NUM_CLASSES = len(torch.unique(torch.cat((y_train, y_val))))
    print(f"감지된 국면 클래스 개수: {NUM_CLASSES}")

    # --- Cost-Sensitive Learning: 클래스 가중치 계산 ---
    class_counts = torch.bincount(y_train)
    # y_train에 없는 클래스가 y_val에 있을 수 있으므로, 최대 클래스 인덱스에 맞춰 크기 조정
    weights = torch.zeros(NUM_CLASSES)
    # bincount 결과가 NUM_CLASSES보다 작을 수 있으므로, 있는 만큼만 채워넣음
    weights[:len(class_counts)] = class_counts.float()
    
    # 0으로 나누는 것을 방지
    weights = 1.0 / (weights + 1e-6) 
    weights = weights / weights.sum() * NUM_CLASSES # 정규화
    weights_tensor = weights.to(device)
    print(f"클래스 가중치: {weights_tensor.cpu().numpy()}")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_features = X_train.shape[2]
    model = TradingTransformer(
        num_features=num_features,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, 
        num_classes=NUM_CLASSES
    ).to(device)

    print(f"\n--- 모델 구조 (총 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}) ---\n")

    # --- 손실 함수 선택 ---
    # criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0).to(device)
    print(f"사용 손실 함수: {criterion.__class__.__name__}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=7) # F1 Score 기준
    scaler = torch.cuda.amp.GradScaler()

    best_val_f1 = 0.0
    early_stopping_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [학습]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
            train_pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_f1, _, _, _ = calculate_metrics(all_train_labels, all_train_preds, NUM_CLASSES)
        
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_preds = [], []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [검증]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1, val_prec, val_recall, val_cm = calculate_metrics(all_val_labels, all_val_preds, NUM_CLASSES)
        
        scheduler.step(val_f1)

        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} (P: {val_prec:.3f}, R: {val_recall:.3f})")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(MODEL_SAVE_DIR, 'best_model_regime.pth')
            torch.save(model.state_dict(), model_path)
            print(f"** 최고 검증 F1 Score 갱신! 모델 저장. (F1: {best_val_f1:.4f}) **")
            print("Validation Confusion Matrix:\n", val_cm)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n! {EARLY_STOPPING_PATIENCE} 에포크 동안 검증 F1 Score가 개선되지 않아 학습을 조기 종료합니다.")
            break
        print(f"EarlyStopping Counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}\n")

    print("학습 완료!")

if __name__ == '__main__':
    main()
