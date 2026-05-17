from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "wfa_optimized_params_output" / "phase53_active_live_shadow"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _gate_map(paper_review: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not paper_review:
        return {}
    return {str(row.get("name")): row for row in paper_review.get("gates", [])}


def _gate_pass(gates: dict[str, dict[str, Any]], name: str) -> bool | None:
    gate = gates.get(name)
    if gate is None:
        return None
    return bool(gate.get("pass"))


def _count_passes(gates: dict[str, dict[str, Any]]) -> dict[str, int]:
    total = len(gates)
    passed = sum(1 for gate in gates.values() if gate.get("pass") is True)
    return {"passed": passed, "failed": total - passed, "total": total}


def _parse_run_stamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _hours_since(run_utc: Any) -> float | None:
    parsed = _parse_run_stamp(run_utc)
    if parsed is None:
        return None
    return round((datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0, 6)


def _status_from_decision(
    *,
    evidence: dict[str, Any] | None,
    paper_review: dict[str, Any] | None,
) -> str:
    paper_status = (paper_review or {}).get("decision", {}).get("status")
    evidence_status = (evidence or {}).get("decision", {}).get("status")
    if paper_status == "READY_FOR_MANUAL_PAPER_REVIEW":
        return "READY_FOR_MANUAL_PAPER_REVIEW"
    if paper_status == "HOLD_PAPER_REVIEW":
        return "ACTIVE_SHADOW_HOLD"
    if evidence_status == "SHADOW_EVIDENCE_READY":
        return "SHADOW_EVIDENCE_READY_NEEDS_PAPER_GATE"
    return "INCOMPLETE_OR_UNKNOWN"


def _build_monitoring(
    *,
    latest_cycle: dict[str, Any],
    evidence_shortfalls: dict[str, Any],
    evidence_projection: dict[str, Any],
    paper_decision: dict[str, Any],
    paper_candidate_export: dict[str, Any],
    latest_history: dict[str, Any] | None,
    stale_after_hours: float,
) -> dict[str, Any]:
    hours = _hours_since((latest_history or {}).get("run_utc"))
    automation_health = "NO_RUN_HISTORY"
    if hours is not None:
        automation_health = "RECENT" if hours <= stale_after_hours else "STALE"

    alerts: list[dict[str, Any]] = []
    if latest_history is None:
        alerts.append({
            "level": "warning",
            "code": "NO_RUN_HISTORY",
            "message": "No active shadow wrapper run history has been recorded yet.",
        })
    elif automation_health == "STALE":
        alerts.append({
            "level": "warning",
            "code": "SHADOW_AUTOMATION_STALE",
            "message": f"Latest wrapper run is older than {stale_after_hours} hours.",
            "hours_since_latest_run": hours,
        })

    if latest_cycle.get("funding_status") != "ok":
        alerts.append({
            "level": "warning",
            "code": "FUNDING_COVERAGE_NOT_OK",
            "message": "Funding coverage is not ok; paper-review promotion must stay blocked.",
            "funding_status": latest_cycle.get("funding_status"),
        })

    if paper_decision.get("status") == "READY_FOR_MANUAL_PAPER_REVIEW":
        if paper_candidate_export.get("candidate_exists"):
            alerts.append({
                "level": "action",
                "code": "MANUAL_PAPER_REVIEW_REQUIRED",
                "message": "Manual-review-only paper candidate exists. Human review is required before any paper order workflow.",
                "candidate_out": paper_candidate_export.get("candidate_out"),
            })
        else:
            alerts.append({
                "level": "critical",
                "code": "READY_BUT_CANDIDATE_MISSING",
                "message": "Paper-review gate is ready but the manual-review candidate manifest is missing.",
                "candidate_out": paper_candidate_export.get("candidate_out"),
            })
    else:
        alerts.append({
            "level": "info",
            "code": "LIVE_EVIDENCE_PENDING",
            "message": "Paper review remains on hold until live shadow evidence gates pass.",
            "failed_gates": paper_decision.get("failed_gates", []),
            "shortfalls": {
                name: evidence_shortfalls.get(name)
                for name in ("observed_days", "unique_signal_times", "total_events", "accepted_events")
            },
        })

    if evidence_projection.get("observed_days_gate_is_time_blocked"):
        alerts.append({
            "level": "info",
            "code": "OBSERVED_DAYS_TIME_BLOCKED",
            "message": "Observed-days evidence cannot pass until enough calendar time has elapsed.",
            "earliest_signal_time": evidence_projection.get("earliest_signal_time_for_observed_days"),
        })
    if evidence_projection.get("accepted_event_gate_is_signal_blocked"):
        alerts.append({
            "level": "info",
            "code": "ACCEPTED_ENTRY_SIGNAL_REQUIRED",
            "message": "At least one live accepted entry event is still required; this cannot be guaranteed by schedule alone.",
            "additional_accepted_events_required": evidence_projection.get("additional_accepted_events_required"),
        })
    if latest_cycle.get("new_events") == 0 and int(latest_cycle.get("duplicate_events") or 0) > 0:
        alerts.append({
            "level": "info",
            "code": "DUPLICATE_POLL_NO_NEW_SIGNAL_TIME",
            "message": "Latest wrapper run saw an already-recorded signal time and correctly avoided duplicate evidence.",
            "duplicate_events": latest_cycle.get("duplicate_events"),
        })
    if paper_candidate_export.get("status") == "SKIPPED_NOT_READY" and paper_candidate_export.get("candidate_exists") is False:
        alerts.append({
            "level": "info",
            "code": "PAPER_CANDIDATE_EXPORT_SKIPPED_AS_DESIGNED",
            "message": "No manual paper candidate exists while the paper-review gate is still HOLD.",
        })

    return {
        "latest_run_utc": (latest_history or {}).get("run_utc"),
        "hours_since_latest_run": hours,
        "stale_after_hours": stale_after_hours,
        "automation_health": automation_health,
        "alerts": alerts,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = _resolve(args.out_dir)
    cycle_path = _resolve(args.cycle_summary)
    evidence_path = _resolve(args.evidence_json)
    paper_path = _resolve(args.paper_review_json)
    events_path = out_dir / "portfolio_shadow_events.jsonl"
    history_index_path = out_dir / "run_history" / "index.jsonl"
    candidate_status_path = out_dir / "paper_candidate_export_status.json"
    candidate_manifest_path = out_dir / "paper_candidate_manifest.json"

    cycle = _load_json_if_exists(cycle_path)
    evidence = _load_json_if_exists(evidence_path)
    paper_review = _load_json_if_exists(paper_path)
    candidate_status = _load_json_if_exists(candidate_status_path)
    events = _load_jsonl(events_path)
    history = _load_jsonl(history_index_path)

    gates = _gate_map(paper_review)
    latest_poll = None
    if cycle and cycle.get("polls"):
        latest_poll = cycle["polls"][-1]

    event_signal_times = sorted({str(row.get("signal_time")) for row in events if row.get("signal_time")})
    accepted_events = [row for row in events if row.get("status") == "accepted"]
    no_entry_events = [row for row in events if row.get("status") == "no_entry_event"]
    event_modes = sorted({str(row.get("observation_mode")) for row in events if row.get("observation_mode")})
    manifest_hashes = sorted({str(row.get("manifest_sha256")) for row in events if row.get("manifest_sha256")})

    gate_groups = {
        "execution_contract": [
            "manifest_execution_assumptions_strict",
            "validation_execution_assumptions_match_manifest",
        ],
        "cpu_cuda_diff": [
            "validation_source_cpu_cuda_diff_pass",
            "validation_source_cpu_cuda_diff_details_match",
        ],
        "portfolio_diagnostic": [
            "manifest_portfolio_diagnostic_strict",
            "source_portfolio_diagnostic_strict",
            "portfolio_diagnostic_source_matches_manifest",
        ],
        "live_evidence_integrity": [
            "cycle_is_live_observation",
            "cycle_manifest_hash_matches",
            "cycle_target_after_train_end",
            "evidence_manifest_hash_matches",
            "evidence_hash_integrity",
            "evidence_after_train_end",
            "evidence_live_only",
            "shadow_evidence_ready",
        ],
        "paper_lock": [
            "paper_automation_disabled",
        ],
    }
    grouped_gate_status = {
        group: {name: _gate_pass(gates, name) for name in names}
        for group, names in gate_groups.items()
    }

    evidence_readiness = (evidence or {}).get("readiness", {})
    paper_decision = (paper_review or {}).get("decision", {})
    cycle_totals = (cycle or {}).get("event_totals", {})
    latest_history = history[-1] if history else None
    latest_cycle = {
        "observation_mode": (cycle or {}).get("observation_mode"),
        "target_signal_time": (latest_poll or {}).get("target_signal_time"),
        "as_of": (latest_poll or {}).get("as_of"),
        "new_events": cycle_totals.get("new"),
        "duplicate_events": cycle_totals.get("duplicates"),
        "accepted_components": cycle_totals.get("accepted_components"),
        "funding_status": ((cycle or {}).get("funding_coverage") or {}).get("status"),
        "funding_lag_hours": ((cycle or {}).get("funding_coverage") or {}).get("lag_hours"),
    }
    paper_candidate_export = {
        "status": (candidate_status or {}).get("status"),
        "candidate_exists": candidate_manifest_path.exists(),
        "candidate_out": (candidate_status or {}).get("candidate_out"),
        "candidate_out_md": (candidate_status or {}).get("candidate_out_md"),
        "cleared_stale_paths": (candidate_status or {}).get("cleared_stale_paths", []),
    }
    monitoring = _build_monitoring(
        latest_cycle=latest_cycle,
        evidence_shortfalls=evidence_readiness.get("shortfalls", {}),
        evidence_projection=evidence_readiness.get("projection", {}),
        paper_decision=paper_decision,
        paper_candidate_export=paper_candidate_export,
        latest_history=latest_history,
        stale_after_hours=args.stale_after_hours,
    )

    report = {
        "schema_version": 1,
        "mode": "phase53_active_shadow_status",
        "status": _status_from_decision(evidence=evidence, paper_review=paper_review),
        "active_paths": {
            "out_dir": str(out_dir),
            "cycle_summary": str(cycle_path),
            "evidence_json": str(evidence_path),
            "paper_review_json": str(paper_path),
            "events_jsonl": str(events_path),
            "run_history_index": str(history_index_path),
            "paper_candidate_export_status": str(candidate_status_path),
            "paper_candidate_manifest": str(candidate_manifest_path),
        },
        "latest_cycle": latest_cycle,
        "stored_evidence": {
            "total_events": len(events),
            "accepted_events": len(accepted_events),
            "no_entry_events": len(no_entry_events),
            "unique_signal_times": len(event_signal_times),
            "first_signal_time": event_signal_times[0] if event_signal_times else None,
            "last_signal_time": event_signal_times[-1] if event_signal_times else None,
            "observation_modes": event_modes,
            "manifest_hashes": manifest_hashes,
        },
        "evidence_decision": (evidence or {}).get("decision"),
        "evidence_shortfalls": evidence_readiness.get("shortfalls", {}),
        "evidence_projection": evidence_readiness.get("projection", {}),
        "paper_review_decision": paper_decision,
        "paper_candidate_export": paper_candidate_export,
        "paper_gate_counts": _count_passes(gates),
        "paper_failed_gates": paper_decision.get("failed_gates", []),
        "grouped_gate_status": grouped_gate_status,
        "run_history": {
            "run_count": len(history),
            "latest": latest_history,
        },
        "monitoring": monitoring,
        "next_action": evidence_readiness.get("next_action", "continue_live_no_order_shadow_logging_and_recheck_evidence"),
    }
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    latest = report["latest_cycle"]
    stored = report["stored_evidence"]
    projection = report.get("evidence_projection", {})
    shortfalls = report.get("evidence_shortfalls", {})
    gate_counts = report.get("paper_gate_counts", {})
    lines = [
        "# Phase53 Active Shadow Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Paper-review decision: `{report.get('paper_review_decision', {}).get('status')}`",
        f"- Paper failed gates: `{report.get('paper_failed_gates')}`",
        f"- Paper candidate export: `{report.get('paper_candidate_export', {}).get('status')}`",
        f"- Automation health: `{report.get('monitoring', {}).get('automation_health')}`",
        f"- Next action: `{report.get('next_action')}`",
        "",
        "## Latest Cycle",
        "",
        f"- Observation mode: `{latest.get('observation_mode')}`",
        f"- Target signal time: `{latest.get('target_signal_time')}`",
        f"- As of: `{latest.get('as_of')}`",
        f"- New events: `{latest.get('new_events')}`",
        f"- Duplicates: `{latest.get('duplicate_events')}`",
        f"- Accepted components: `{latest.get('accepted_components')}`",
        f"- Funding: `{latest.get('funding_status')}` lag `{latest.get('funding_lag_hours')}` hours",
        "",
        "## Stored Evidence",
        "",
        f"- Total events: `{stored.get('total_events')}`",
        f"- Unique signal times: `{stored.get('unique_signal_times')}`",
        f"- Accepted events: `{stored.get('accepted_events')}`",
        f"- No-entry events: `{stored.get('no_entry_events')}`",
        f"- First signal time: `{stored.get('first_signal_time')}`",
        f"- Last signal time: `{stored.get('last_signal_time')}`",
        f"- Observation modes: `{stored.get('observation_modes')}`",
        "",
        "## Remaining Evidence",
        "",
    ]
    for name in ("observed_days", "unique_signal_times", "total_events", "accepted_events"):
        item = shortfalls.get(name, {})
        lines.append(
            f"- {name}: observed `{_fmt(item.get('observed'))}`, threshold `{_fmt(item.get('threshold'))}`, remaining `{_fmt(item.get('remaining'))}` {_fmt(item.get('unit'))}"
        )
    lines.extend(
        [
            "",
            "## Projection",
            "",
            f"- Additional unique signal times: `{projection.get('additional_unique_signal_times_required')}`",
            f"- Additional full live polls for total events: `{projection.get('additional_full_live_polls_for_total_events')}`",
            f"- Additional accepted events: `{projection.get('additional_accepted_events_required')}`",
            f"- Earliest observed-days signal time: `{projection.get('earliest_signal_time_for_observed_days')}`",
            "",
            "## Gate Groups",
            "",
            f"- Paper gates: `{gate_counts.get('passed')}/{gate_counts.get('total')}` passed",
        ]
    )
    for group, statuses in report.get("grouped_gate_status", {}).items():
        status_text = ", ".join(f"{name}={value}" for name, value in statuses.items())
        lines.append(f"- {group}: `{status_text}`")
    lines.extend(["", "## Monitoring", ""])
    monitoring = report.get("monitoring", {})
    lines.append(f"- Latest run UTC: `{monitoring.get('latest_run_utc')}`")
    lines.append(f"- Hours since latest run: `{monitoring.get('hours_since_latest_run')}`")
    lines.append(f"- Automation health: `{monitoring.get('automation_health')}`")
    lines.append("")
    for alert in monitoring.get("alerts", []):
        lines.append(f"- `{alert.get('level')}` `{alert.get('code')}`: {alert.get('message')}")
    latest_history = report.get("run_history", {}).get("latest")
    lines.extend(
        [
            "",
            "## Paper Candidate",
            "",
            f"- Export status: `{report.get('paper_candidate_export', {}).get('status')}`",
            f"- Candidate exists: `{str(report.get('paper_candidate_export', {}).get('candidate_exists')).lower()}`",
            f"- Candidate path: `{report.get('paper_candidate_export', {}).get('candidate_out')}`",
            "",
            "## Run History",
            "",
            f"- Run count: `{report.get('run_history', {}).get('run_count')}`",
            f"- Latest run: `{latest_history}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase53 active live shadow evidence and paper-review status.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cycle-summary", type=Path, default=DEFAULT_OUT_DIR / "latest_cycle_summary.json")
    parser.add_argument("--evidence-json", type=Path, default=DEFAULT_OUT_DIR / "latest_evidence.json")
    parser.add_argument("--paper-review-json", type=Path, default=DEFAULT_OUT_DIR / "paper_review_gate.json")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_DIR / "status_report.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_DIR / "status_report.md")
    parser.add_argument("--stale-after-hours", type=float, default=3.0)
    args = parser.parse_args()

    report = summarize(args)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "target_signal_time": report["latest_cycle"]["target_signal_time"],
                "total_events": report["stored_evidence"]["total_events"],
                "unique_signal_times": report["stored_evidence"]["unique_signal_times"],
                "accepted_events": report["stored_evidence"]["accepted_events"],
                "paper_review_status": report["paper_review_decision"].get("status"),
                "paper_failed_gates": report["paper_failed_gates"],
                "out_json": str(out_json),
                "out_md": str(out_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
