import torch
import torch.nn as nn
import numpy as np

# --- 모델 정의 (V2: [CLS] 토큰 적용 및 구조 개선) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5001): # sequence_length + 1
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TradingTransformerV2(nn.Module):
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
        nn.init.normal_(self.cls_token, std=0.02) # [CLS] 토큰 초기화

    def forward(self, src):
        # src shape: (batch_size, seq_len, num_features)
        src = self.encoder_input(src) * np.sqrt(self.d_model)
        
        # ✨ V2 핵심: [CLS] 토큰을 모든 시퀀스의 맨 앞에 추가
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # ✨ V2 핵심: [CLS] 토큰의 최종 출력값만 사용하여 분류
        cls_output = output[:, 0, :]
        
        output = self.decoder(cls_output)
        return output