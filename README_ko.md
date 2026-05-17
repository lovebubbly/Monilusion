# Monilusion

Monilusion은 BTCUSDT Binance USD-M 선물 전략을 연구, 검증, 섀도우 관찰, 실행 준비까지 다루는 작업공간입니다. 현재 구조는 크게 세 부분으로 나뉩니다.

- `real_M1.py`: 기본값이 안전한 실거래/테스트넷 봇 골격
- `emacrossmart.py`, `tools/`: CUDA 기반 전략 탐색과 검증 도구
- Phase53 파이프라인: 실시간 섀도우 증거와 수동 paper-review gate

현재 기본 운영 원칙은 보수적입니다. 검증하고, 섀도우로 관찰하고, 리뷰한 다음에 paper/live 승격을 판단합니다. 어떤 백테스트 결과도 실자금 거래 허가로 해석하면 안 됩니다.

## 안전 기본값

- 실제 비밀값은 `.env`에만 둡니다. `.env.example`은 커밋하고 `.env`는 절대 커밋하지 않습니다.
- `real_M1.py`는 기본적으로 Binance testnet과 `DRY_RUN=true`로 실행됩니다.
- 실계정 주문 실행은 아래 세 조건이 모두 필요합니다.
  - `BINANCE_USE_TESTNET=false`
  - `DRY_RUN=false`
  - `ALLOW_LIVE_TRADING=YES_I_UNDERSTAND`
- 생성 데이터, 모델 바이너리, 로그, 마이그레이션 번들, numpy/pickle/joblib/torch 산출물은 Git에서 제외합니다.
- Phase53 파이프라인은 live shadow evidence gate가 통과되기 전까지 paper/live 승격을 보류합니다.

보안 체크리스트는 [SECURITY.md](SECURITY.md)를 참고하세요.

## 주요 진입점

| 경로 | 용도 |
| --- | --- |
| `real_M1.py` | 안전 기본값이 적용된 실거래/테스트넷 봇 골격 |
| `emacrossmart.py` | CUDA BTCUSDT 1h EMA 전략 탐색 및 백테스트 |
| `src/v2/` | 실행 가정 검증용 CPU reference backtest 구성요소 |
| `tools/run_phase53_active_shadow_cycle.py` | 크로스플랫폼 Phase53 섀도우 사이클 오케스트레이터 |
| `tools/run_phase53_active_shadow_cycle.ps1` | Windows용 Phase53 실행 래퍼 |
| `tools/run_phase53_active_shadow_cycle.sh` | macOS/Linux용 Phase53 실행 래퍼 |
| `tools/create_phase53_migration_bundle.py` | Mac 이관용 번들 생성 |
| `docs/phase53_mac_mini_migration.md` | Mac mini 이관 절차서 |

## 설치

가능하면 Python 3.11을 사용하세요.

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install python-binance pandas numpy python-dotenv pandas_ta schedule numba cupy-cuda12x
pip install -r requirements-phase53-shadow.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install python-binance pandas numpy python-dotenv pandas_ta schedule numba
pip install -r requirements-phase53-shadow.txt
```

CUDA 가속 탐색은 Windows/NVIDIA PC에서 돌리는 것이 맞습니다. Mac mini M2는 장시간 안정적으로 섀도우 증거를 수집하는 역할에 더 적합합니다.

## 설정

`.env.example`을 복사해 `.env`를 만들고 필요한 값만 채우세요.

실행 봇의 안전 기본값:

```text
BINANCE_USE_TESTNET=true
DRY_RUN=true
ALLOW_LIVE_TRADING=
```

CUDA 탐색용 주요 변수:

```text
OFFLINE_OHLCV_H1=C:\Monilusion\data\BTCUSDT_1h.csv
SEARCH_PROFILE=smoke
PARAM_LIMIT=1
BT_START_DATE=2026-03-01
BT_END_DATE=2026-05-01
CUDA_BATCH_SIZE=1
WFA_OUTPUT_DIR=wfa_optimized_params_output\emacrossmart_smoke
```

## CUDA 전략 탐색

Windows에서 오프라인 스모크 실행:

```powershell
$env:OFFLINE_OHLCV_H1='C:\Monilusion\data\BTCUSDT_1h.csv'
$env:SEARCH_PROFILE='smoke'
$env:PARAM_LIMIT='1'
$env:BT_START_DATE='2026-03-01'
$env:BT_END_DATE='2026-05-01'
$env:CUDA_BATCH_SIZE='1'
$env:WFA_OUTPUT_DIR='wfa_optimized_params_output\emacrossmart_smoke'
.\venv\Scripts\python.exe emacrossmart.py
```

`emacrossmart.py`는 오프라인 데이터가 없을 때만 Binance에서 과거 klines를 가져옵니다. 주문을 생성하지 않습니다.

## Phase53 섀도우 사이클

네트워크 업데이트 없이 스모크 실행:

```powershell
.\venv\Scripts\python.exe tools\run_phase53_active_shadow_cycle.py --skip-updates
```

Windows 래퍼:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_phase53_active_shadow_cycle.ps1 --skip-updates
```

macOS/Linux 래퍼:

```bash
bash tools/run_phase53_active_shadow_cycle.sh --skip-updates
```

충분한 live evidence가 쌓이기 전 정상적인 보류 상태는 아래와 같습니다.

```text
ACTIVE_SHADOW_HOLD
HOLD_PAPER_REVIEW
```

## Mac Mini 이관

Windows에서 이관 번들을 생성합니다.

```powershell
.\venv\Scripts\python.exe tools\create_phase53_migration_bundle.py --out migration_bundles\phase53_mac_migration_bundle.zip
```

이후 [docs/phase53_mac_mini_migration.md](docs/phase53_mac_mini_migration.md)를 따라가면 됩니다. 번들은 생성 데이터와 런타임 증거를 포함하므로 Git에 올리지 않습니다.

## 검증

최근 변경에 사용한 빠른 로컬 체크:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\venv\Scripts\python.exe tools\run_phase53_active_shadow_cycle.py --skip-updates
```

`pytest`가 설치되어 있다면:

```powershell
.\venv\Scripts\python.exe -m pytest -q src\tests
```

## 저장소 구조

| 경로 | 설명 |
| --- | --- |
| `src/` | 핵심 Python 모듈 |
| `src/v2/` | reference backtest 및 실행 가정 검증 모듈 |
| `src/tests/` | 회귀 테스트 |
| `tools/` | 연구, 검증, 섀도우, 이관 유틸리티 |
| `docs/` | 운영 절차서 |
| `legacy/` | 오래된 실험 코드. 검토 없이 확장하지 않는 편이 좋습니다. |
| `data/`, `saved_models/`, `cache/`, `wfa_optimized_params_output/` | 로컬/생성 산출물. 소스가 아닙니다. |

## 현재 상태

Phase53 후보는 아직 자동 paper trading 대상이 아닙니다. live shadow evidence readiness가 통과되고 paper-review gate가 명시적으로 준비될 때까지 no-order shadow logging 상태를 유지해야 합니다.
