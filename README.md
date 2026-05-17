# Monilusion

Monilusion is a BTCUSDT Binance USD-M futures research and execution workspace. The repository now separates three concerns:

- safe-by-default live bot scaffolding in `real_M1.py`
- CUDA-accelerated strategy research in `emacrossmart.py` and `tools/`
- Phase53 shadow-evidence and paper-review gates before any paper/live promotion

The current operating posture is conservative: validate, shadow, review, then decide. Do not treat any strategy output as permission to trade real funds.

## Safety Defaults

- Secrets live in `.env`; commit `.env.example`, never `.env`.
- `real_M1.py` defaults to Binance testnet and `DRY_RUN=true`.
- Real-account order execution requires all of:
  - `BINANCE_USE_TESTNET=false`
  - `DRY_RUN=false`
  - `ALLOW_LIVE_TRADING=YES_I_UNDERSTAND`
- Generated data, model binaries, logs, migration bundles, numpy arrays, pickles, joblib files, and torch checkpoints are ignored.
- The Phase53 pipeline currently holds paper/live promotion until live shadow evidence gates pass.

See [SECURITY.md](SECURITY.md) for the security checklist.

## Main Entry Points

| Path | Purpose |
| --- | --- |
| `real_M1.py` | Safe-by-default live/testnet trading bot scaffold |
| `emacrossmart.py` | CUDA BTCUSDT 1h EMA strategy search and backtest runner |
| `src/v2/` | CPU reference backtest components for execution-assumption checks |
| `tools/run_phase53_active_shadow_cycle.py` | Cross-platform Phase53 shadow cycle orchestrator |
| `tools/run_phase53_active_shadow_cycle.ps1` | Windows wrapper for the Phase53 cycle |
| `tools/run_phase53_active_shadow_cycle.sh` | macOS/Linux wrapper for the Phase53 cycle |
| `tools/create_phase53_migration_bundle.py` | Builds the Mac migration artifact bundle |
| `docs/phase53_mac_mini_migration.md` | Mac mini migration runbook |

## Setup

Use Python 3.11 if possible.

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

CUDA acceleration is only expected on the Windows/NVIDIA box. The Mac mini M2 is better suited for stable long-running shadow evidence collection, not CUDA search.

## Configuration

Create `.env` from `.env.example` and fill only the values you need.

Minimum safe live-bot defaults:

```text
BINANCE_USE_TESTNET=true
DRY_RUN=true
ALLOW_LIVE_TRADING=
```

Useful CUDA research variables:

```text
OFFLINE_OHLCV_H1=C:\Monilusion\data\BTCUSDT_1h.csv
SEARCH_PROFILE=smoke
PARAM_LIMIT=1
BT_START_DATE=2026-03-01
BT_END_DATE=2026-05-01
CUDA_BATCH_SIZE=1
WFA_OUTPUT_DIR=wfa_optimized_params_output\emacrossmart_smoke
```

## CUDA Strategy Search

Quick offline smoke on Windows:

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

`emacrossmart.py` uses Binance only for historical klines when offline data is not provided. It does not place orders.

## Phase53 Shadow Cycle

Run a no-network smoke cycle:

```powershell
.\venv\Scripts\python.exe tools\run_phase53_active_shadow_cycle.py --skip-updates
```

Windows wrapper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_phase53_active_shadow_cycle.ps1 --skip-updates
```

macOS/Linux wrapper:

```bash
bash tools/run_phase53_active_shadow_cycle.sh --skip-updates
```

The expected hold state before enough live evidence is:

```text
ACTIVE_SHADOW_HOLD
HOLD_PAPER_REVIEW
```

## Mac Mini Migration

Build the transfer bundle on Windows:

```powershell
.\venv\Scripts\python.exe tools\create_phase53_migration_bundle.py --out migration_bundles\phase53_mac_migration_bundle.zip
```

Then follow [docs/phase53_mac_mini_migration.md](docs/phase53_mac_mini_migration.md). The bundle is intentionally ignored by Git because it contains generated data and runtime evidence.

## Validation

Fast local checks used for recent changes:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\venv\Scripts\python.exe tools\run_phase53_active_shadow_cycle.py --skip-updates
```

If `pytest` is installed:

```powershell
.\venv\Scripts\python.exe -m pytest -q src\tests
```

## Repository Layout

| Path | Notes |
| --- | --- |
| `src/` | Core Python modules |
| `src/v2/` | Reference backtest and execution-assumption modules |
| `src/tests/` | Focused regression tests |
| `tools/` | Research, validation, shadow, and migration utilities |
| `docs/` | Operational runbooks |
| `legacy/` | Archived experiments; avoid extending without review |
| `data/`, `saved_models/`, `cache/`, `wfa_optimized_params_output/` | Local/generated artifacts, not source |

## Current Status

The Phase53 candidate is not automatically paper-tradable yet. It remains blocked on live shadow evidence readiness and should stay in no-order shadow logging until the paper-review gate is explicitly ready.
