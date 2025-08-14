# 바이낸스 선물 자동매매 봇

이 프로젝트는 바이낸스 선물 시장, 특히 BTCUSDT 페어를 위한 정교한 자동매매 시스템을 구현합니다. 고급 머신러닝 기술과 견고한 거래 전략, 포괄적인 위험 관리를 통합합니다.

## 주요 기능

*   **자동 거래:** 사전 정의된 전략과 실시간 시장 데이터를 기반으로 거래를 자동으로 실행합니다.
*   **시장 국면 분류:** PyTorch 기반의 트랜스포머 모델을 활용하여 시장 상황(예: 추세장, 횡보장, 변동성장)을 분류하고, 이를 통해 적응형 거래 결정을 가능하게 합니다.
*   **동적 전략 및 필터:** 여러 진입 전 필터(국면, 시간, 서킷 브레이커)로 강화된 EMA 교차 전략을 사용하여 견고성을 향상시킵니다.
*   **고급 위험 관리:** 지능적인 포지션 규모 조정, 자동 손절매(SL) 및 이익실현(TP) 주문, 하이브리드 청산 프로토콜(시간 기반 정지, ATR 트레일링 스탑)을 포함합니다.
*   **데이터 관리:** 바이낸스에서 과거 K-라인 데이터를 가져와 SQLite 데이터베이스에 저장합니다.
*   **운영 모니터링:** 중요한 이벤트에 대한 광범위한 로깅 및 이메일 알림 기능을 제공합니다.

## 사용 기술

*   **Python:** 핵심 프로그래밍 언어.
*   **Binance API (`python-binance`):** 실시간 거래소 상호작용을 위해 사용됩니다.
*   **Pandas & NumPy:** 데이터 조작 및 수치 연산을 위한 필수 라이브러리.
*   **Pandas-TA:** 기술 분석 지표 계산을 위한 강력한 라이브러리.
*   **PyTorch:** 딥러닝 모델 개발(시장 국면 분류기)을 위해 사용됩니다.
*   **SQLite:** 과거 데이터 저장을 위해 사용됩니다.
*   **`python-dotenv`:** 환경 변수 보안 관리를 위해 사용됩니다.
*   **`schedule`:** 자동화된 작업 스케줄링을 위해 사용됩니다.

## 시작하기

### 전제 조건

Python 3.x 및 `pip`이 설치되어 있는지 확인하십시오.

### 설치

pip을 사용하여 필요한 Python 라이브러리를 설치하십시오:

```bash
pip install python-binance pandas numpy pandas_ta torch scikit-learn python-dotenv schedule tqdm seaborn matplotlib
```

### 설정

1.  **`.env` 파일 생성:** 프로젝트의 루트 디렉토리(`C:\Monilusion\`)에 `.env`라는 파일을 생성하십시오.
2.  **자격 증명 추가:** `.env` 파일에 바이낸스 API 키와 이메일 알림 설정을 입력하십시오. **이 파일을 버전 제어 시스템에 커밋하지 마십시오.**

    ```
    BINANCE_API_KEY=YOUR_BINANCE_API_KEY
    BINANCE_SECRET_KEY=YOUR_BINANCE_SECRET_KEY
    EMAIL_SENDER_ADDRESS=your_email@example.com
    EMAIL_SENDER_PASSWORD=your_email_password # 또는 앱별 비밀번호
    EMAIL_RECEIVER_ADDRESS=recipient_email@example.com
    SMTP_SERVER=smtp.example.com
    SMTP_PORT=587
    ```

## 사용법

### 자동매매 봇 실행

주요 자동매매 봇 로직은 `real_M1.py`에 있습니다. 이 봇은 핵심 거래 작업을 주기적으로(예: 시간별) 스케줄링하여 지속적으로 실행되도록 설계되었습니다.

자동매매 봇을 시작하려면:

```bash
python real_M1.py
```

봇은 `live_trading_bot_phase1.log`에 활동을 기록하고, 설정된 경우 중요한 이벤트에 대한 이메일 알림을 보냅니다.

### 시장 국면 분류 모델 학습

트랜스포머 기반 시장 국면 분류 모델은 독립적으로 학습될 수 있습니다. 이를 위해서는 `data/processed_data_regime_final` 디렉토리에 전처리된 데이터가 있어야 합니다.

모델을 학습하려면:

```bash
python src/train_model.py
```

학습된 모델은 `saved_models_regime_transformer_v2` 디렉토리에 저장됩니다.

## 프로젝트 구조

*   `src/`: 거래 시스템의 핵심 모듈(데이터 파이프라인, 특징 공학, 모델, 필터 등)을 포함합니다.
*   `data/`: 과거 시장 데이터 및 전처리된 데이터셋을 저장합니다.
*   `saved_models/`: 학습된 머신러닝 모델의 위치입니다.
*   `legacy/`: 오래되거나 실험적인 스크립트를 포함합니다.
*   `cudare.py`: 전략 최적화를 위한 GPU 가속 백테스팅 스크립트입니다.
*   `real_M1.py`: 실시간 거래 봇 구현체입니다.

## 라이선스

(선택 사항: 여기에 라이선스 정보를 추가하십시오)
