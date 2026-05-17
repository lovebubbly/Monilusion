# Phase53 Session Handoff

Use this document to give a fresh Codex session the missing project context. It captures the decisions made in the Windows session before the Mac mini migration.

## Paste This Into A Fresh Codex Session

```text
You are working in the Monilusion repo after the Phase53 Mac mini migration.

Context:
- The project is a BTCUSDT Binance USD-M futures research and shadow-validation workspace.
- Do not place orders. Current execution permission is NO_ORDERS_SHADOW_LOGGING_ONLY.
- Keep paper automation on HOLD until the paper-review gate explicitly becomes ready.
- The live shadow evidence collector was migrated from the Windows RTX PC to the Mac mini for stability.
- The Windows PC remains useful for CUDA-heavy strategy searches. The Mac mini is the preferred machine for long-running live shadow evidence collection.

Important commits already pushed:
- 7108b68 chore: harden trading pipeline and shadow validation
- 1f92186 docs: document CUDA research workflow

Current expected status before enough live evidence:
- ACTIVE_SHADOW_HOLD
- HOLD_PAPER_REVIEW
- SHADOW_EVIDENCE_INSUFFICIENT
- BLOCKED_LIVE_SHADOW_EVIDENCE

Do not mark the strategy goal complete unless the live evidence and paper-review gates prove it. Do not start automated paper trading just because the code exists.

Primary files to read first:
- README.md
- SECURITY.md
- docs/phase53_mac_mini_migration.md
- docs/phase53_session_handoff.md
- wfa_optimized_params_output/phase53_active_live_shadow/status_report.md
- wfa_optimized_params_output/phase53_active_live_shadow/goal_completion_matrix.md

Normal Mac smoke command:
bash tools/run_phase53_active_shadow_cycle.sh --skip-updates

Normal live shadow cycle command:
bash tools/run_phase53_active_shadow_cycle.sh

If a result looks promising, inspect paper_review_gate.json and status_report.md first. A paper_candidate_manifest is only a manual-review artifact, not permission for automated orders.
```

## Source Of Truth

| Concern | Source of truth |
| --- | --- |
| Code and docs | GitHub `master` |
| Live shadow evidence after migration | Mac mini checkout |
| CUDA search and heavy GPU validation | Windows RTX PC |
| Secrets | Local `.env` only, never Git |
| Migration bundle | External zip, never Git |

Do not run the same hourly live shadow automation on both Windows and Mac at the same time. Duplicate pollers can create confusing evidence history even when duplicate events are filtered.

## Current Evidence State From Windows Handoff

Last Windows status snapshot before/around migration:

- Status: `ACTIVE_SHADOW_HOLD`
- Paper-review decision: `HOLD_PAPER_REVIEW`
- Failed paper gate: `shadow_evidence_ready`
- Paper candidate export: `SKIPPED_NOT_READY`
- Total events: `12`
- Unique signal times: `2`
- Accepted events: `0`
- Remaining observed days at that snapshot: `13.875`
- Earliest observed-days signal time at that snapshot: `2026-05-31T07:00:00+00:00`

The exact counts on Mac may now be newer. Always trust the Mac runtime files after migration.

## Gate Meaning

`ACTIVE_SHADOW_HOLD` is a good state while evidence is still short. It means the collector is running but promotion is blocked.

`HOLD_PAPER_REVIEW` is also expected until:

- enough calendar time has elapsed,
- enough unique live signal times have been observed,
- enough live event rows exist,
- at least one accepted live entry event exists,
- manifest hashes and observation modes remain clean,
- CPU/CUDA and execution-assumption checks remain consistent.

The most important rule: no paper/live execution should happen merely because backtests were good.

## Files That Must Not Be Committed

- `.env`
- `.env.*`
- `data/`
- `saved_models/`
- `cache/`
- `wfa_optimized_params_output/`
- `migration_bundles/`
- logs, sqlite databases, numpy arrays, pickles, joblib files, torch checkpoints

`.env.example` is intentionally committed because it contains only placeholders and safe defaults.

## Windows Return Path

You only need to return to the Windows RTX PC when:

- a new CUDA-heavy search is needed,
- a large parameter sweep must be run,
- a CPU/CUDA reference diff needs Windows GPU reproduction,
- the Mac evidence points to a strategy variant worth deeper research.

For ordinary shadow status review, keep working on the Mac.

If you transfer evidence back to Windows, transfer the runtime artifact directory out of band. Do not commit `wfa_optimized_params_output/`.

## Quick Review Checklist

Before treating the Mac run as healthy:

1. `bash tools/run_phase53_active_shadow_cycle.sh --skip-updates` succeeds.
2. `status_report.md` shows recent automation health.
3. `paper_review_gate.json` still reports `HOLD_PAPER_REVIEW` until evidence is actually ready.
4. `paper_candidate_manifest.json` does not exist while the paper gate is on hold.
5. No `.env` or runtime artifacts are staged in Git.
6. Only one machine is running the live shadow polling schedule.
