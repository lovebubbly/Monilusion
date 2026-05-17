from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from run_live_portfolio_shadow_poll import NO_ORDER_PERMISSION, manifest_fingerprint, validate_portfolio_shadow_manifest  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    out = Path(path)
    return out if out.is_absolute() else base / out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _parse_event_time(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _gate(name: str, passed: bool, observed: Any, threshold: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _positive_remaining(threshold: float, observed: float) -> float:
    return round(max(0.0, float(threshold) - float(observed)), 6)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return int(max(0, numerator + denominator - 1) // denominator)


def evaluate_shadow_evidence(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve(args.portfolio_manifest)
    events_path = _resolve(args.events_jsonl)
    manifest = _load_json(manifest_path)
    validate_portfolio_shadow_manifest(manifest)
    expected_manifest_sha256 = manifest_fingerprint(manifest_path)
    events = _load_jsonl(events_path)
    event_ids = [str(row.get("event_id")) for row in events if row.get("event_id")]
    unique_event_ids = set(event_ids)
    signal_times = [_parse_event_time(row.get("signal_time")) for row in events]
    signal_times = [ts for ts in signal_times if ts is not None]
    first_signal = min(signal_times) if signal_times else None
    last_signal = max(signal_times) if signal_times else None
    unique_signal_times = sorted({ts.isoformat() for ts in signal_times})
    observed_days = 0.0
    if first_signal is not None and last_signal is not None:
        observed_days = max(0.0, (last_signal - first_signal).total_seconds() / 86400.0)

    component_ids = {str(row.get("component_id")) for row in manifest.get("components", [])}
    observed_components = {str(row.get("component_id")) for row in events if row.get("component_id")}
    missing_components = sorted(component_ids - observed_components)
    accepted_events = [row for row in events if row.get("status") == "accepted"]
    no_entry_events = [row for row in events if row.get("status") == "no_entry_event"]
    duplicate_event_count = max(0, len(event_ids) - len(unique_event_ids))
    event_manifest_hashes = sorted({str(row.get("manifest_sha256")) for row in events if row.get("manifest_sha256")})
    missing_manifest_hash_count = sum(1 for row in events if not row.get("manifest_sha256"))
    mismatched_manifest_hash_count = sum(
        1 for row in events if row.get("manifest_sha256") and row.get("manifest_sha256") != expected_manifest_sha256
    )
    event_observation_modes = sorted({str(row.get("observation_mode")) for row in events if row.get("observation_mode")})
    missing_observation_mode_count = sum(1 for row in events if not row.get("observation_mode"))
    non_live_evidence_count = sum(1 for row in events if row.get("live_evidence_eligible") is not True)

    gates = [
        _gate(
            "manifest_no_order_only",
            manifest.get("execution_permission") == NO_ORDER_PERMISSION and manifest.get("ready_for_paper") is not True,
            manifest.get("execution_permission"),
            f"== {NO_ORDER_PERMISSION} and ready_for_paper != true",
            "Manifest must remain no-order shadow only.",
        ),
        _gate(
            "observed_days",
            observed_days >= args.min_observed_days,
            round(observed_days, 6),
            f">= {args.min_observed_days}",
            "Calendar span covered by logged shadow evidence.",
        ),
        _gate(
            "unique_signal_times",
            len(unique_signal_times) >= args.min_unique_signal_times,
            len(unique_signal_times),
            f">= {args.min_unique_signal_times}",
            "Distinct signal timestamps logged by the shadow poller.",
        ),
        _gate(
            "total_events",
            len(events) >= args.min_total_events,
            len(events),
            f">= {args.min_total_events}",
            "Total no-order component events in the evidence log.",
        ),
        _gate(
            "accepted_events",
            len(accepted_events) >= args.min_accepted_events,
            len(accepted_events),
            f">= {args.min_accepted_events}",
            "Accepted component entry events observed in shadow logs.",
        ),
        _gate(
            "duplicate_events",
            duplicate_event_count <= args.max_duplicate_events,
            duplicate_event_count,
            f"<= {args.max_duplicate_events}",
            "Duplicate event IDs in the JSONL evidence log.",
        ),
        _gate(
            "component_coverage",
            len(missing_components) <= args.max_missing_components,
            missing_components,
            f"missing <= {args.max_missing_components}",
            "Every portfolio component should have shadow evidence unless explicitly allowed.",
        ),
        _gate(
            "manifest_hash_present",
            missing_manifest_hash_count <= args.max_missing_manifest_hash_events,
            missing_manifest_hash_count,
            f"<= {args.max_missing_manifest_hash_events}",
            "Every event should record the source manifest hash unless legacy events are explicitly allowed.",
        ),
        _gate(
            "manifest_hash_match",
            mismatched_manifest_hash_count <= args.max_mismatched_manifest_hash_events,
            {"mismatched": mismatched_manifest_hash_count, "hashes": event_manifest_hashes},
            f"mismatched <= {args.max_mismatched_manifest_hash_events}",
            "Events must belong to the manifest currently being evaluated.",
        ),
        _gate(
            "observation_mode_present",
            missing_observation_mode_count <= args.max_missing_observation_mode_events,
            missing_observation_mode_count,
            f"<= {args.max_missing_observation_mode_events}",
            "Every event should record whether it came from live observation or historical replay.",
        ),
        _gate(
            "live_evidence_only",
            non_live_evidence_count <= args.max_non_live_evidence_events,
            {"non_live": non_live_evidence_count, "modes": event_observation_modes},
            f"non_live <= {args.max_non_live_evidence_events}",
            "Manual paper-review evidence must exclude historical holdout replay events.",
        ),
    ]
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    ready_for_review = not failed
    shortfalls = {
        "observed_days": {
            "observed": round(observed_days, 6),
            "threshold": args.min_observed_days,
            "remaining": _positive_remaining(args.min_observed_days, observed_days),
            "unit": "days",
        },
        "unique_signal_times": {
            "observed": len(unique_signal_times),
            "threshold": args.min_unique_signal_times,
            "remaining": int(max(0, args.min_unique_signal_times - len(unique_signal_times))),
            "unit": "signal_times",
        },
        "total_events": {
            "observed": len(events),
            "threshold": args.min_total_events,
            "remaining": int(max(0, args.min_total_events - len(events))),
            "unit": "events",
        },
        "accepted_events": {
            "observed": len(accepted_events),
            "threshold": args.min_accepted_events,
            "remaining": int(max(0, args.min_accepted_events - len(accepted_events))),
            "unit": "accepted_events",
        },
        "duplicate_events": {
            "observed": duplicate_event_count,
            "threshold": args.max_duplicate_events,
            "excess": int(max(0, duplicate_event_count - args.max_duplicate_events)),
            "unit": "duplicates",
        },
        "missing_components": {
            "observed": len(missing_components),
            "threshold": args.max_missing_components,
            "excess": int(max(0, len(missing_components) - args.max_missing_components)),
            "components": missing_components,
        },
        "missing_manifest_hash_events": {
            "observed": missing_manifest_hash_count,
            "threshold": args.max_missing_manifest_hash_events,
            "excess": int(max(0, missing_manifest_hash_count - args.max_missing_manifest_hash_events)),
            "unit": "events",
        },
        "mismatched_manifest_hash_events": {
            "observed": mismatched_manifest_hash_count,
            "threshold": args.max_mismatched_manifest_hash_events,
            "excess": int(max(0, mismatched_manifest_hash_count - args.max_mismatched_manifest_hash_events)),
            "event_hashes": event_manifest_hashes,
        },
        "missing_observation_mode_events": {
            "observed": missing_observation_mode_count,
            "threshold": args.max_missing_observation_mode_events,
            "excess": int(max(0, missing_observation_mode_count - args.max_missing_observation_mode_events)),
            "unit": "events",
        },
        "non_live_evidence_events": {
            "observed": non_live_evidence_count,
            "threshold": args.max_non_live_evidence_events,
            "excess": int(max(0, non_live_evidence_count - args.max_non_live_evidence_events)),
            "event_observation_modes": event_observation_modes,
        },
    }
    component_count = len(component_ids)
    total_events_remaining = int(shortfalls["total_events"]["remaining"])
    unique_signal_times_remaining = int(shortfalls["unique_signal_times"]["remaining"])
    additional_full_polls_for_events = _ceil_div(total_events_remaining, component_count)
    earliest_signal_time_for_observed_days = None
    if first_signal is not None:
        earliest_signal_time_for_observed_days = (
            first_signal + pd.Timedelta(days=float(args.min_observed_days))
        ).isoformat()
    projection = {
        "events_per_full_live_poll": component_count,
        "additional_unique_signal_times_required": unique_signal_times_remaining,
        "additional_full_live_polls_for_total_events": additional_full_polls_for_events,
        "additional_accepted_events_required": int(shortfalls["accepted_events"]["remaining"]),
        "earliest_signal_time_for_observed_days": earliest_signal_time_for_observed_days,
        "observed_days_gate_is_time_blocked": shortfalls["observed_days"]["remaining"] > 0,
        "accepted_event_gate_is_signal_blocked": shortfalls["accepted_events"]["remaining"] > 0,
        "notes": [
            "Assumes each future live poll writes one component event per active manifest component.",
            "Accepted-entry evidence cannot be guaranteed by schedule; it requires the strategy to emit a live entry signal.",
        ],
    }
    return {
        "schema_version": 1,
        "mode": "portfolio_shadow_evidence_gate",
        "source_manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "events_jsonl": str(events_path),
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "summary": {
            "event_count": len(events),
            "unique_event_count": len(unique_event_ids),
            "duplicate_event_count": duplicate_event_count,
            "accepted_event_count": len(accepted_events),
            "no_entry_event_count": len(no_entry_events),
            "component_count": len(component_ids),
            "observed_component_count": len(observed_components),
            "missing_components": missing_components,
            "event_manifest_hashes": event_manifest_hashes,
            "missing_manifest_hash_count": missing_manifest_hash_count,
            "mismatched_manifest_hash_count": mismatched_manifest_hash_count,
            "event_observation_modes": event_observation_modes,
            "missing_observation_mode_count": missing_observation_mode_count,
            "non_live_evidence_count": non_live_evidence_count,
            "unique_signal_time_count": len(unique_signal_times),
            "first_signal_time": first_signal.isoformat() if first_signal is not None else None,
            "last_signal_time": last_signal.isoformat() if last_signal is not None else None,
            "observed_days": round(observed_days, 6),
        },
        "criteria": {
            "min_observed_days": args.min_observed_days,
            "min_unique_signal_times": args.min_unique_signal_times,
            "min_total_events": args.min_total_events,
            "min_accepted_events": args.min_accepted_events,
            "max_duplicate_events": args.max_duplicate_events,
            "max_missing_components": args.max_missing_components,
            "max_missing_manifest_hash_events": args.max_missing_manifest_hash_events,
            "max_mismatched_manifest_hash_events": args.max_mismatched_manifest_hash_events,
            "max_missing_observation_mode_events": args.max_missing_observation_mode_events,
            "max_non_live_evidence_events": args.max_non_live_evidence_events,
        },
        "readiness": {
            "manual_paper_review": ready_for_review,
            "paper_trading_automation": "HOLD",
            "blocking_gates": failed,
            "shortfalls": shortfalls,
            "projection": projection,
            "next_action": (
                "manual_review_allowed_but_keep_paper_automation_disabled"
                if ready_for_review
                else "continue_live_no_order_shadow_logging_and_recheck_evidence"
            ),
        },
        "gates": gates,
        "decision": {
            "status": "SHADOW_EVIDENCE_READY_FOR_MANUAL_PAPER_REVIEW" if ready_for_review else "SHADOW_EVIDENCE_INSUFFICIENT",
            "ready_for_manual_paper_review": ready_for_review,
            "ready_for_paper": False,
            "failed_gates": failed,
            "rationale": (
                "Shadow evidence coverage gates passed; paper automation remains disabled pending manual review."
                if ready_for_review
                else "Shadow evidence is not yet sufficient for manual paper review; keep paper automation disabled."
            ),
        },
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Portfolio Shadow Evidence Gate",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Ready for manual paper review: `{str(result['decision']['ready_for_manual_paper_review']).lower()}`",
        f"- Ready for paper: `false`",
        f"- Manifest SHA-256: `{result.get('manifest_sha256')}`",
        f"- Events: `{summary['event_count']}`",
        f"- Accepted events: `{summary['accepted_event_count']}`",
        f"- Unique signal times: `{summary['unique_signal_time_count']}`",
        f"- Observed days: `{summary['observed_days']}`",
        f"- Next action: `{result.get('readiness', {}).get('next_action')}`",
        "",
        "| gate | pass | observed | threshold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in result["gates"]:
        lines.append(f"| {gate['name']} | {str(gate['pass']).lower()} | `{gate['observed']}` | `{gate['threshold']}` |")
    lines.extend(["", "## Remaining Evidence", ""])
    shortfalls = result.get("readiness", {}).get("shortfalls", {})
    for name in ("observed_days", "unique_signal_times", "total_events", "accepted_events"):
        item = shortfalls.get(name, {})
        lines.append(
            f"- {name}: observed `{item.get('observed')}`, threshold `{item.get('threshold')}`, remaining `{item.get('remaining')}` {item.get('unit', '')}"
        )
    for name in (
        "duplicate_events",
        "missing_components",
        "missing_manifest_hash_events",
        "mismatched_manifest_hash_events",
        "missing_observation_mode_events",
        "non_live_evidence_events",
    ):
        item = shortfalls.get(name, {})
        lines.append(f"- {name}: observed `{item.get('observed')}`, allowed `{item.get('threshold')}`, excess `{item.get('excess')}`")
    projection = result.get("readiness", {}).get("projection", {})
    lines.extend(
        [
            "",
            "## Projection",
            "",
            f"- Events per full live poll: `{projection.get('events_per_full_live_poll')}`",
            f"- Additional unique signal times required: `{projection.get('additional_unique_signal_times_required')}`",
            f"- Additional full live polls for total events: `{projection.get('additional_full_live_polls_for_total_events')}`",
            f"- Additional accepted events required: `{projection.get('additional_accepted_events_required')}`",
            f"- Earliest observed-days signal time: `{projection.get('earliest_signal_time_for_observed_days')}`",
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether no-order portfolio shadow logs are sufficient for paper review.")
    parser.add_argument("--portfolio-manifest", required=True, type=Path)
    parser.add_argument("--events-jsonl", required=True, type=Path)
    parser.add_argument("--min-observed-days", type=float, default=14.0)
    parser.add_argument("--min-unique-signal-times", type=int, default=24)
    parser.add_argument("--min-total-events", type=int, default=84)
    parser.add_argument("--min-accepted-events", type=int, default=1)
    parser.add_argument("--max-duplicate-events", type=int, default=0)
    parser.add_argument("--max-missing-components", type=int, default=0)
    parser.add_argument("--max-missing-manifest-hash-events", type=int, default=0)
    parser.add_argument("--max-mismatched-manifest-hash-events", type=int, default=0)
    parser.add_argument("--max-missing-observation-mode-events", type=int, default=0)
    parser.add_argument("--max-non-live-evidence-events", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_shadow_evidence(args)
    if args.out_json:
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out_md = _resolve(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, result)
    print(
        json.dumps(
            {
                "decision": result["decision"]["status"],
                "ready_for_manual_paper_review": result["decision"]["ready_for_manual_paper_review"],
                "ready_for_paper": result["decision"]["ready_for_paper"],
                "event_count": result["summary"]["event_count"],
                "accepted_event_count": result["summary"]["accepted_event_count"],
                "observed_days": result["summary"]["observed_days"],
                "failed_gates": result["decision"]["failed_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
