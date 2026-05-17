from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


NO_ORDER_PERMISSION = "NO_ORDERS_SHADOW_LOGGING_ONLY"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_time(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True).tz_convert(None)


def _rank_summary(rank: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    signal_times = sorted({_parse_time(row.get("signal_time")) for row in events if row.get("signal_time")})
    signal_times = [ts for ts in signal_times if ts is not None]
    accepted = [row for row in events if row.get("status") == "accepted"]
    rejected = [row for row in events if row.get("status") == "rejected"]
    no_entry = [row for row in events if row.get("status") == "no_entry_event"]
    span_days = 0.0
    if len(signal_times) >= 2:
        span_days = (signal_times[-1] - signal_times[0]).total_seconds() / 86400.0
    return {
        "rank": rank,
        "events": len(events),
        "unique_signal_times": len(signal_times),
        "shadow_span_days": round(span_days, 4),
        "accepted_signals": len(accepted),
        "rejected_signals": len(rejected),
        "no_entry_events": len(no_entry),
        "first_signal_time": signal_times[0].isoformat() if signal_times else None,
        "last_signal_time": signal_times[-1].isoformat() if signal_times else None,
    }


def _candidate_diff_pass(candidate: dict[str, Any]) -> bool:
    return candidate.get("source_period_diff_pass", candidate.get("source_period_cpu_gpu_diff_pass", False)) is True


def _modern_validation_failures(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(manifest.get("schema_version", 0)) < 2:
        failures.append("manifest_schema_version_below_2")
    assumptions = manifest.get("assumptions", {})
    if assumptions.get("intrabar_policy_base") != "conservative":
        failures.append("manifest_missing_conservative_intrabar_base")
    if assumptions.get("intrabar_policy_comparison") != "optimistic":
        failures.append("manifest_missing_optimistic_intrabar_comparison")
    pbo = manifest.get("topk_cscv_pbo", {})
    if pbo.get("enabled") is not True:
        failures.append("manifest_purged_cpcv_not_enabled")
    if int(pbo.get("purge_bars") or 0) <= 0:
        failures.append("manifest_missing_purged_cpcv_purge_bars")
    if int(pbo.get("embargo_bars") or 0) <= 0:
        failures.append("manifest_missing_purged_cpcv_embargo_bars")
    if int(pbo.get("samples") or 0) <= 0:
        failures.append("manifest_missing_purged_cpcv_samples")

    for candidate in manifest.get("candidates", []):
        rank = candidate.get("rank")
        decision = candidate.get("decision", {})
        if decision.get("status") != "PROMOTE_TO_SHADOW":
            failures.append(f"rank_{rank}_not_promoted_to_shadow")
        if decision.get("ready_for_shadow") is not True:
            failures.append(f"rank_{rank}_not_ready_for_shadow")
        if decision.get("failed_gates"):
            failures.append(f"rank_{rank}_has_failed_gates")
        if not _candidate_diff_pass(candidate):
            failures.append(f"rank_{rank}_source_diff_not_passed")
        if not candidate.get("intrabar_policy_band"):
            failures.append(f"rank_{rank}_missing_intrabar_policy_band")
    return failures


def evaluate_gate(
    *,
    manifest_path: Path,
    events_path: Path,
    min_shadow_days: float,
    min_unique_signal_times: int,
    min_accepted_signals: int,
    extended_oos_path: Path | None = None,
    min_post_source_pnl_pct: float = 0.0,
    min_post_source_pf: float = 1.0,
    require_modern_validation: bool = True,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    events = _load_jsonl(events_path)
    rank_to_events: dict[int, list[dict[str, Any]]] = {}
    for row in events:
        rank_to_events.setdefault(int(row["rank"]), []).append(row)

    manifest_candidates = [int(row["rank"]) for row in manifest.get("candidates", [])]
    all_permissions_ok = all(row.get("execution_permission") == NO_ORDER_PERMISSION for row in events)
    all_modes_ok = all(row.get("mode") == "live_shadow_poll" for row in events)
    all_known_candidates = all(rank in manifest_candidates for rank in rank_to_events)

    candidate_summaries = [_rank_summary(rank, rank_to_events.get(rank, [])) for rank in manifest_candidates]
    failures: list[str] = []
    if manifest.get("decision") != "PROMOTE_TO_SHADOW":
        failures.append("manifest_not_promoted_to_shadow")
    if manifest.get("ready_for_shadow") is not True:
        failures.append("manifest_not_ready_for_shadow")
    if manifest.get("execution_permission") != NO_ORDER_PERMISSION:
        failures.append("manifest_permission_not_no_order")
    if manifest.get("ready_for_paper") is True:
        failures.append("manifest_already_marks_paper_ready_without_gate")
    if not manifest_candidates:
        failures.append("manifest_has_no_candidates")
    if not events:
        failures.append("no_live_shadow_events")
    if not all_permissions_ok:
        failures.append("event_permission_not_no_order")
    if not all_modes_ok:
        failures.append("events_not_from_live_shadow_poll")
    if not all_known_candidates:
        failures.append("events_contain_unknown_candidate_rank")
    if require_modern_validation:
        failures.extend(_modern_validation_failures(manifest))

    for row in candidate_summaries:
        rank = row["rank"]
        if row["unique_signal_times"] < min_unique_signal_times:
            failures.append(f"rank_{rank}_unique_signal_times_below_{min_unique_signal_times}")
        if row["shadow_span_days"] < min_shadow_days:
            failures.append(f"rank_{rank}_shadow_span_days_below_{min_shadow_days}")
        if row["accepted_signals"] < min_accepted_signals:
            failures.append(f"rank_{rank}_accepted_signals_below_{min_accepted_signals}")

    extended_oos = None
    if extended_oos_path is not None:
        extended_oos = _load_json(extended_oos_path)
        for row in extended_oos.get("candidates", []):
            rank = int(row["rank"])
            post = row.get("post_source_period", {})
            pnl_pct = float(post.get("net_pnl_pct_initial", 0.0))
            pf_value = post.get("net_profit_factor", 0.0)
            post_pf = float("inf") if pf_value == "inf" else float(pf_value)
            if pnl_pct < min_post_source_pnl_pct:
                failures.append(f"rank_{rank}_post_source_pnl_pct_below_{min_post_source_pnl_pct}")
            if post_pf < min_post_source_pf:
                failures.append(f"rank_{rank}_post_source_pf_below_{min_post_source_pf}")

    ready_for_manual_paper_review = not failures
    return {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_live_shadow_events": str(events_path),
        "source_extended_oos": str(extended_oos_path) if extended_oos_path is not None else None,
        "decision": "READY_FOR_MANUAL_PAPER_REVIEW" if ready_for_manual_paper_review else "HOLD_PAPER",
        "ready_for_manual_paper_review": ready_for_manual_paper_review,
        "ready_for_automated_paper": False,
        "paper_trading_automation": "HOLD",
        "criteria": {
            "min_shadow_days": min_shadow_days,
            "min_unique_signal_times": min_unique_signal_times,
            "min_accepted_signals": min_accepted_signals,
            "min_post_source_pnl_pct": min_post_source_pnl_pct,
            "min_post_source_pf": min_post_source_pf,
            "require_modern_validation": require_modern_validation,
            "required_shadow_manifest_schema_version": 2 if require_modern_validation else None,
            "required_intrabar_policy_base": "conservative" if require_modern_validation else None,
            "required_intrabar_policy_comparison": "optimistic" if require_modern_validation else None,
            "required_pbo": "purged_embargoed_cpcv" if require_modern_validation else None,
            "required_event_mode": "live_shadow_poll",
            "required_execution_permission": NO_ORDER_PERMISSION,
        },
        "failure_reasons": sorted(set(failures)),
        "candidate_summaries": candidate_summaries,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Shadow To Paper Gate",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Ready for manual paper review: `{str(result['ready_for_manual_paper_review']).lower()}`",
        f"- Ready for automated paper: `{str(result['ready_for_automated_paper']).lower()}`",
        f"- Paper trading automation: `{result['paper_trading_automation']}`",
        "",
        "## Criteria",
    ]
    for key, value in result["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failure Reasons"])
    if result["failure_reasons"]:
        lines.extend(f"- `{reason}`" for reason in result["failure_reasons"])
    else:
        lines.append("- _None._")
    lines.extend(
        [
            "",
            "## Candidate Summaries",
            "| rank | events | unique signal times | span days | accepted | rejected | no entry |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["candidate_summaries"]:
        lines.append(
            f"| {row['rank']} | {row['events']} | {row['unique_signal_times']} | "
            f"{row['shadow_span_days']} | {row['accepted_signals']} | "
            f"{row['rejected_signals']} | {row['no_entry_events']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether live shadow logs are sufficient for paper review.")
    parser.add_argument(
        "--shadow-manifest",
        type=Path,
        default=Path("wfa_optimized_params_output/phase14_shadow_candidates_2019_2025.json"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("wfa_optimized_params_output/live_shadow_phase14/live_shadow_events.jsonl"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("wfa_optimized_params_output/paper_gate_phase14_live_shadow.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("wfa_optimized_params_output/paper_gate_phase14_live_shadow.md"),
    )
    parser.add_argument("--min-shadow-days", type=float, default=14.0)
    parser.add_argument("--min-unique-signal-times", type=int, default=20)
    parser.add_argument("--min-accepted-signals", type=int, default=3)
    parser.add_argument("--extended-oos", type=Path, default=None)
    parser.add_argument("--min-post-source-pnl-pct", type=float, default=0.0)
    parser.add_argument("--min-post-source-pf", type=float, default=1.0)
    parser.add_argument("--allow-legacy-manifest", action="store_true", help="Do not require schema v2 intrabar-band and purged-CPCV evidence.")
    args = parser.parse_args()

    result = evaluate_gate(
        manifest_path=args.shadow_manifest if args.shadow_manifest.is_absolute() else Path.cwd() / args.shadow_manifest,
        events_path=args.events if args.events.is_absolute() else Path.cwd() / args.events,
        min_shadow_days=args.min_shadow_days,
        min_unique_signal_times=args.min_unique_signal_times,
        min_accepted_signals=args.min_accepted_signals,
        extended_oos_path=(
            args.extended_oos if args.extended_oos is not None and args.extended_oos.is_absolute()
            else (Path.cwd() / args.extended_oos if args.extended_oos is not None else None)
        ),
        min_post_source_pnl_pct=args.min_post_source_pnl_pct,
        min_post_source_pf=args.min_post_source_pf,
        require_modern_validation=not args.allow_legacy_manifest,
    )
    out_json = args.out_json if args.out_json.is_absolute() else Path.cwd() / args.out_json
    out_md = args.out_md if args.out_md.is_absolute() else Path.cwd() / args.out_md
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, result)
    print(json.dumps({
        "decision": result["decision"],
        "ready_for_manual_paper_review": result["ready_for_manual_paper_review"],
        "ready_for_automated_paper": result["ready_for_automated_paper"],
        "failure_count": len(result["failure_reasons"]),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
