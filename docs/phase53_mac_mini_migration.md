# Phase53 Mac Mini Migration Runbook

This runbook moves the BTCUSDT 1H Phase53 no-order live shadow evidence collector from the Windows workstation to a more stable Mac mini.

For a fresh Codex session on the Mac, read `docs/phase53_session_handoff.md` first. It contains the missing conversation context, the current gate posture, and the prompt to paste into a new session.

## What Moves Through Git

Commit and push source code, wrappers, tests, and docs:

- `tools/run_phase53_active_shadow_cycle.py`
- `tools/run_phase53_active_shadow_cycle.sh`
- `tools/run_phase53_active_shadow_cycle.ps1`
- `tools/rebind_phase53_paths.py`
- `tools/create_phase53_migration_bundle.py`
- `tools/summarize_phase53_shadow_status.py`
- `tools/audit_phase53_goal_completion.py`
- `tools/export_portfolio_paper_candidate_manifest.py`
- `src/tests/`
- `requirements-phase53-shadow.txt`
- this document

Do not commit `.env`, `data/`, `wfa_optimized_params_output/`, or `migration_bundles/`.

## What Moves Outside Git

The active shadow state is ignored by Git and must be transferred as an artifact bundle:

- `data/BTCUSDT_1h.csv`
- `wfa_optimized_params_output/futures_context/BTCUSDT_funding_rate_8h_20190101_20260516.csv`
- `wfa_optimized_params_output/phase53_active_portfolio_shadow_manifest.json`
- `wfa_optimized_params_output/phase53_active_portfolio_shadow_registry.json`
- `wfa_optimized_params_output/phase53_current_train_refresh_20260517_active_registry_from_existing/`
- `wfa_optimized_params_output/phase53_active_live_shadow/`
- `wfa_optimized_params_output/wfo_portfolio_phase53_dsr_meta2_full_n1771_20260517.json`

Create the bundle on Windows:

```powershell
.\venv\Scripts\python.exe tools\create_phase53_migration_bundle.py
```

The command writes a zip under `migration_bundles/` and prints the bundle SHA-256.

## Mac Setup

Clone the repo on the Mac mini, then extract the migration zip at the repo root so paths such as `data/BTCUSDT_1h.csv` and `wfa_optimized_params_output/...` appear under the checkout.

Use Python 3.11 if possible:

```bash
cd /path/to/Monilusion
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-phase53-shadow.txt
```

If `python3.11` is not installed, install it first with Homebrew or pyenv. The pinned `pandas-ta` version is old, so Python 3.11 is the least surprising path.

## Rebind Windows Paths

The active artifacts were produced on Windows and contain `C:\Monilusion\...` paths. Rebind them to the Mac checkout:

```bash
source .venv/bin/activate
python tools/rebind_phase53_paths.py --new-root "$PWD" --write
```

Dry-run first if desired:

```bash
python tools/rebind_phase53_paths.py --new-root "$PWD"
```

## Smoke Test

Run one no-order shadow cycle manually:

```bash
chmod +x tools/run_phase53_active_shadow_cycle.sh
./tools/run_phase53_active_shadow_cycle.sh
```

If you only want to verify local wiring before allowing network updates, run:

```bash
./tools/run_phase53_active_shadow_cycle.sh --skip-updates
```

Expected current behavior while evidence is still short:

- `evidence_decision`: `SHADOW_EVIDENCE_INSUFFICIENT`
- `paper_review_decision`: `HOLD_PAPER_REVIEW`
- `paper_candidate_export_status.json`: `SKIPPED_NOT_READY`
- `status_report.json`: `ACTIVE_SHADOW_HOLD`
- `goal_completion_matrix.json`: `BLOCKED_LIVE_SHADOW_EVIDENCE`

The wrapper never places orders. It preserves `NO_ORDERS_SHADOW_LOGGING_ONLY` and keeps paper automation on `HOLD`.

## Launchd Hourly Schedule

Create `~/Library/LaunchAgents/com.monilusion.phase53-shadow.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.monilusion.phase53-shadow</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/path/to/Monilusion/tools/run_phase53_active_shadow_cycle.sh</string>
  </array>
  <key>StartInterval</key>
  <integer>3600</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>/path/to/Monilusion</string>
  <key>StandardOutPath</key>
  <string>/path/to/Monilusion/wfa_optimized_params_output/phase53_active_live_shadow/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>/path/to/Monilusion/wfa_optimized_params_output/phase53_active_live_shadow/launchd.err.log</string>
</dict>
</plist>
```

Replace `/path/to/Monilusion`, then load it:

```bash
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.monilusion.phase53-shadow.plist
launchctl kickstart -k "gui/$(id -u)/com.monilusion.phase53-shadow"
```

Check status:

```bash
launchctl print "gui/$(id -u)/com.monilusion.phase53-shadow"
tail -n 80 wfa_optimized_params_output/phase53_active_live_shadow/launchd.out.log
tail -n 80 wfa_optimized_params_output/phase53_active_live_shadow/launchd.err.log
```

Unload if needed:

```bash
launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.monilusion.phase53-shadow.plist
```

## Handoff Checklist

Before turning off the Windows automation:

1. Mac manual wrapper run succeeds.
2. `status_report.json` shows `automation_health=RECENT`.
3. `goal_completion_matrix.json` still shows either `BLOCKED_LIVE_SHADOW_EVIDENCE` or, later, `COMPLETE`.
4. `portfolio_shadow_events.jsonl` contains the Windows events plus any new Mac events.
5. `paper_candidate_manifest.json` does not exist while the gate is `HOLD_PAPER_REVIEW`.
6. Mac launchd has run at least once after `RunAtLoad`.

After that, pause or delete the Windows Codex automation to avoid duplicate hourly polling from two machines.
