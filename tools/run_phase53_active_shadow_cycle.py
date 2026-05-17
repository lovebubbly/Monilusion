from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "wfa_optimized_params_output" / "phase53_active_live_shadow"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _append_history_index(out_dir: Path, run_dir: Path, stamp: str) -> None:
    cycle = _load_json(out_dir / "latest_cycle_summary.json")
    evidence = _load_json(out_dir / "latest_evidence.json")
    paper = _load_json(out_dir / "paper_review_gate.json")
    polls = list(cycle.get("polls", []))
    if not polls:
        raise SystemExit("Cannot archive Phase53 shadow run: latest_cycle_summary.json has no polls.")
    last_poll = polls[-1]
    index_entry = {
        "run_utc": stamp,
        "run_dir": str(run_dir),
        "target_signal_time": last_poll.get("target_signal_time"),
        "as_of": last_poll.get("as_of"),
        "new_events": cycle.get("event_totals", {}).get("new"),
        "duplicate_events": cycle.get("event_totals", {}).get("duplicates"),
        "accepted_components": cycle.get("event_totals", {}).get("accepted_components"),
        "funding_status": (cycle.get("funding_coverage") or {}).get("status"),
        "evidence_status": evidence.get("decision", {}).get("status"),
        "evidence_failed_gates": evidence.get("decision", {}).get("failed_gates"),
        "paper_review_status": paper.get("decision", {}).get("status"),
        "paper_review_failed_gates": paper.get("decision", {}).get("failed_gates"),
    }
    index_path = out_dir / "run_history" / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_entry, ensure_ascii=False) + "\n")


def _new_run_dir(out_dir: Path) -> tuple[Path, str]:
    history_root = out_dir / "run_history"
    history_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = history_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = history_root / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir, stamp


def _copy_existing(out_dir: Path, run_dir: Path, names: list[str]) -> None:
    for name in names:
        source = out_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the active Phase53 no-order shadow cycle and archive status artifacts.")
    parser.add_argument("--rebind-paths", action="store_true", help="Rebind C:\\Monilusion paths to this checkout before running.")
    parser.add_argument("--old-root", default=r"C:\Monilusion")
    parser.add_argument("--skip-updates", action="store_true", help="Skip Binance OHLCV/funding updates for local smoke tests.")
    args, cycle_extra_args = parser.parse_known_args()

    py = sys.executable
    if args.rebind_paths:
        _run([py, str(ROOT / "tools" / "rebind_phase53_paths.py"), "--old-root", args.old_root, "--new-root", str(ROOT), "--write"])

    cycle_cmd = [
        py,
        str(ROOT / "tools" / "run_portfolio_shadow_cycle.py"),
        "--portfolio-manifest", str(ROOT / "wfa_optimized_params_output" / "phase53_active_portfolio_shadow_manifest.json"),
        "--out-dir", str(OUT_DIR),
        "--cycle-summary", str(OUT_DIR / "latest_cycle_summary.json"),
        "--evidence-json", str(OUT_DIR / "latest_evidence.json"),
        "--evidence-md", str(OUT_DIR / "latest_evidence.md"),
        "--run-paper-review-gate",
        "--active-registry", str(ROOT / "wfa_optimized_params_output" / "phase53_active_portfolio_shadow_registry.json"),
        "--validation-json",
        str(
            ROOT
            / "wfa_optimized_params_output"
            / "phase53_current_train_refresh_20260517_active_registry_from_existing"
            / "validation_phase53_portfolio_shadow_current_train_20260517_active_registry_from_existing.json"
        ),
        "--paper-review-json", str(OUT_DIR / "paper_review_gate.json"),
        "--paper-review-md", str(OUT_DIR / "paper_review_gate.md"),
    ]
    if not args.skip_updates:
        cycle_cmd.extend(["--update-data", "--update-funding"])
    cycle_cmd.extend(cycle_extra_args)
    _run(cycle_cmd)

    run_dir, stamp = _new_run_dir(OUT_DIR)
    _copy_existing(
        OUT_DIR,
        run_dir,
        [
            "latest_cycle_summary.json",
            "latest_cycle_summary.md",
            "latest_evidence.json",
            "latest_evidence.md",
            "latest_portfolio_shadow_poll_summary.json",
            "paper_review_gate.json",
            "paper_review_gate.md",
            "portfolio_shadow_state.json",
        ],
    )
    _append_history_index(OUT_DIR, run_dir, stamp)

    _run([
        py,
        str(ROOT / "tools" / "export_portfolio_paper_candidate_manifest.py"),
        "--paper-review-gate", str(OUT_DIR / "paper_review_gate.json"),
        "--out", str(OUT_DIR / "paper_candidate_manifest.json"),
        "--out-md", str(OUT_DIR / "paper_candidate_manifest.md"),
        "--status-out", str(OUT_DIR / "paper_candidate_export_status.json"),
        "--skip-if-not-ready",
        "--clear-stale-on-skip",
    ])
    _run([py, str(ROOT / "tools" / "summarize_phase53_shadow_status.py")])
    _run([py, str(ROOT / "tools" / "audit_phase53_goal_completion.py")])
    _copy_existing(
        OUT_DIR,
        run_dir,
        [
            "goal_completion_matrix.json",
            "goal_completion_matrix.md",
            "paper_candidate_export_status.json",
            "paper_candidate_manifest.json",
            "paper_candidate_manifest.md",
            "status_report.json",
            "status_report.md",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
