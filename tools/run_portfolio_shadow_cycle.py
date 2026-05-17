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

from diff_cuda_cpu_reference import _load_ohlcv  # noqa: E402
from evaluate_portfolio_paper_review_gate import _write_markdown as _write_paper_review_markdown  # noqa: E402
from evaluate_portfolio_paper_review_gate import evaluate_paper_review_gate  # noqa: E402
from evaluate_portfolio_shadow_evidence import _write_markdown as _write_evidence_markdown  # noqa: E402
from evaluate_portfolio_shadow_evidence import evaluate_shadow_evidence  # noqa: E402
from update_binance_funding_rate import update_funding_rates  # noqa: E402
from run_live_portfolio_shadow_poll import run_live_portfolio_shadow_poll  # noqa: E402
from update_binance_futures_ohlcv import _parse_time_ms, update_ohlcv  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


def _signal_times(
    *,
    csv_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_polls: int,
) -> list[pd.Timestamp]:
    df = _load_ohlcv(csv_path, start.isoformat(), end.isoformat())
    times = [ts for ts in df.index if start <= ts <= end]
    if not times:
        raise SystemExit(f"No signal timestamps in requested range: {start.isoformat()} to {end.isoformat()}")
    if max_polls > 0:
        times = times[-max_polls:]
    return times


def _evidence_args(
    *,
    manifest: Path,
    events_jsonl: Path,
    min_observed_days: float,
    min_unique_signal_times: int,
    min_total_events: int,
    min_accepted_events: int,
    max_duplicate_events: int,
    max_missing_components: int,
    max_missing_manifest_hash_events: int,
    max_mismatched_manifest_hash_events: int,
    max_missing_observation_mode_events: int,
    max_non_live_evidence_events: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        portfolio_manifest=manifest,
        events_jsonl=events_jsonl,
        min_observed_days=min_observed_days,
        min_unique_signal_times=min_unique_signal_times,
        min_total_events=min_total_events,
        min_accepted_events=min_accepted_events,
        max_duplicate_events=max_duplicate_events,
        max_missing_components=max_missing_components,
        max_missing_manifest_hash_events=max_missing_manifest_hash_events,
        max_mismatched_manifest_hash_events=max_mismatched_manifest_hash_events,
        max_missing_observation_mode_events=max_missing_observation_mode_events,
        max_non_live_evidence_events=max_non_live_evidence_events,
    )


def _paper_review_args(
    *,
    active_registry: Path,
    manifest: Path,
    validation_json: Path | None,
    cycle_summary: Path,
    evidence_json: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        active_registry=active_registry,
        portfolio_manifest=manifest,
        validation_json=validation_json,
        cycle_summary=cycle_summary,
        evidence_json=evidence_json,
    )


def _write_cycle_markdown(path: Path, summary: dict[str, Any]) -> None:
    paper_gate = summary.get("paper_review_gate", {})
    lines = [
        "# Portfolio Shadow Cycle",
        "",
        f"- Observation mode: `{summary['observation_mode']}`",
        f"- Funding update: `{summary.get('funding_update', {}).get('mode', summary.get('funding_update', {}).get('reason'))}`",
        f"- Poll count: `{summary['poll_count']}`",
        f"- New events: `{summary['event_totals']['new']}`",
        f"- Duplicates: `{summary['event_totals']['duplicates']}`",
        f"- Accepted components: `{summary['event_totals']['accepted_components']}`",
        f"- Funding coverage: `{(summary.get('funding_coverage') or {}).get('status')}`",
        f"- Evidence decision: `{summary['evidence_gate']['decision']}`",
        f"- Evidence next action: `{summary['evidence_gate'].get('next_action')}`",
        f"- Paper-review decision: `{paper_gate.get('decision', paper_gate.get('reason'))}`",
        f"- Effective ready for manual paper review: `{str(summary['effective_ready_for_manual_paper_review']).lower()}`",
        f"- Ready for paper: `false`",
        "",
        "## Remaining Evidence",
        "",
    ]
    shortfalls = summary.get("evidence_gate", {}).get("shortfalls", {})
    for name in ("observed_days", "unique_signal_times", "total_events", "accepted_events"):
        item = shortfalls.get(name, {})
        lines.append(
            f"- {name}: observed `{item.get('observed')}`, threshold `{item.get('threshold')}`, remaining `{item.get('remaining')}` {item.get('unit', '')}"
        )
    for name in ("duplicate_events", "missing_components", "missing_manifest_hash_events", "mismatched_manifest_hash_events"):
        item = shortfalls.get(name, {})
        lines.append(f"- {name}: observed `{item.get('observed')}`, allowed `{item.get('threshold')}`, excess `{item.get('excess')}`")
    if paper_gate and not paper_gate.get("skipped"):
        lines.extend([
            "",
            "## Paper Review Gate",
            "",
            f"- Decision: `{paper_gate.get('decision')}`",
            f"- Ready for manual paper review: `{str(paper_gate.get('ready_for_manual_paper_review')).lower()}`",
            f"- Ready for automated paper: `{str(paper_gate.get('ready_for_automated_paper')).lower()}`",
            f"- Failed gates: `{paper_gate.get('failed_gates')}`",
        ])
    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = _resolve(args.csv)
    manifest_path = _resolve(args.portfolio_manifest)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle_summary_path = _resolve(args.cycle_summary)
    evidence_json = _resolve(args.evidence_json)
    evidence_md = _resolve(args.evidence_md)
    cycle_md = cycle_summary_path.with_suffix(".md")
    manifest = _load_json(manifest_path)

    data_update = {"skipped": True, "reason": "update_data_false"}
    if args.update_data:
        data_update = update_ohlcv(
            csv_path=csv_path,
            symbol=args.symbol,
            interval=args.interval,
            start_ms=_parse_time_ms(args.start),
            end_ms=_parse_time_ms(args.end),
            overlap_bars=args.overlap_bars,
            limit=1000,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            write=True,
            backup=True,
        )

    funding_update = {"skipped": True, "reason": "update_funding_false"}
    if args.update_funding:
        funding_csv = manifest.get("assumptions", {}).get("funding_rate_csv")
        if not funding_csv:
            raise SystemExit("--update-funding requested, but manifest assumptions have no funding_rate_csv.")
        funding_path = Path(funding_csv)
        funding_path = funding_path if funding_path.is_absolute() else ROOT / funding_path
        funding_update = update_funding_rates(
            csv_path=funding_path,
            symbol=args.symbol,
            start_ms=_parse_time_ms(args.funding_start),
            end_ms=_parse_time_ms(args.funding_end or args.end),
            overlap_events=args.funding_overlap_events,
            limit=args.funding_limit,
            timeout=args.funding_timeout,
            sleep_seconds=args.funding_sleep_seconds,
            write=True,
            backup=True,
        )

    entry_delay_bars = int(manifest.get("assumptions", {}).get("entry_delay_bars", 1))
    signal_start = _parse_ts(args.signal_start)
    signal_end = _parse_ts(args.signal_end)
    observation_mode = "latest_live_closed_candle"
    poll_summaries = []

    if signal_start is not None or signal_end is not None:
        if signal_start is None or signal_end is None:
            raise SystemExit("--signal-start and --signal-end must be supplied together.")
        observation_mode = "historical_holdout_replay_not_live_evidence"
        for signal_time in _signal_times(
            csv_path=csv_path,
            start=signal_start,
            end=signal_end,
            max_polls=args.max_polls,
        ):
            as_of = signal_time + pd.Timedelta(hours=entry_delay_bars)
            poll_summaries.append(
                run_live_portfolio_shadow_poll(
                    manifest_path=manifest_path,
                    csv_path=csv_path,
                    out_dir=out_dir,
                    component_ids=set(args.component_id) if args.component_id else None,
                    signal_time=signal_time,
                    as_of=as_of,
                    max_funding_lag_hours=args.max_funding_lag_hours,
                    allow_stale_funding=args.allow_stale_funding,
                    observation_mode=observation_mode,
                    live_evidence_eligible=False,
                )
            )
    else:
        poll_summaries.append(
            run_live_portfolio_shadow_poll(
                manifest_path=manifest_path,
                csv_path=csv_path,
                out_dir=out_dir,
                component_ids=set(args.component_id) if args.component_id else None,
                signal_time=None,
                as_of=_parse_ts(args.as_of),
                max_funding_lag_hours=args.max_funding_lag_hours,
                allow_stale_funding=args.allow_stale_funding,
                observation_mode=observation_mode,
                live_evidence_eligible=True,
            )
        )

    events_jsonl = out_dir / "portfolio_shadow_events.jsonl"
    evidence = evaluate_shadow_evidence(
        _evidence_args(
            manifest=manifest_path,
            events_jsonl=events_jsonl,
            min_observed_days=args.min_observed_days,
            min_unique_signal_times=args.min_unique_signal_times,
            min_total_events=args.min_total_events,
            min_accepted_events=args.min_accepted_events,
            max_duplicate_events=args.max_duplicate_events,
            max_missing_components=args.max_missing_components,
            max_missing_manifest_hash_events=args.max_missing_manifest_hash_events,
            max_mismatched_manifest_hash_events=args.max_mismatched_manifest_hash_events,
            max_missing_observation_mode_events=args.max_missing_observation_mode_events,
            max_non_live_evidence_events=args.max_non_live_evidence_events,
        )
    )
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_evidence_markdown(evidence_md, evidence)

    raw_review_ready = evidence["decision"]["ready_for_manual_paper_review"]
    effective_review_ready = raw_review_ready and observation_mode == "latest_live_closed_candle"
    if raw_review_ready and not effective_review_ready:
        effective_reason = "Historical holdout replay can validate plumbing but cannot by itself satisfy live paper-review evidence."
    else:
        effective_reason = evidence["decision"]["rationale"]

    summary = {
        "schema_version": 1,
        "mode": "portfolio_shadow_cycle",
        "observation_mode": observation_mode,
        "data_update": data_update,
        "funding_update": funding_update,
        "poll_count": len(poll_summaries),
        "polls": poll_summaries,
        "event_totals": {
            "new": sum(int(row.get("new_event_count", 0)) for row in poll_summaries),
            "duplicates": sum(int(row.get("duplicate_event_count", 0)) for row in poll_summaries),
            "accepted_components": sum(int(row.get("accepted_component_count", 0)) for row in poll_summaries),
        },
        "funding_coverage": poll_summaries[-1].get("funding_coverage") if poll_summaries else None,
        "evidence_gate": {
            "decision": evidence["decision"]["status"],
            "ready_for_manual_paper_review": raw_review_ready,
            "ready_for_paper": evidence["decision"]["ready_for_paper"],
            "failed_gates": evidence["decision"]["failed_gates"],
            "next_action": evidence.get("readiness", {}).get("next_action"),
            "shortfalls": evidence.get("readiness", {}).get("shortfalls", {}),
            "projection": evidence.get("readiness", {}).get("projection", {}),
            "out_json": str(evidence_json),
            "out_md": str(evidence_md),
        },
        "effective_ready_for_manual_paper_review": effective_review_ready,
        "effective_paper_review_rationale": effective_reason,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "paper_review_gate": {"skipped": True, "reason": "run_paper_review_gate_false"},
        "outputs": {
            "out_dir": str(out_dir),
            "events_jsonl": str(events_jsonl),
            "events_csv": str(out_dir / "portfolio_shadow_events.csv"),
            "state": str(out_dir / "portfolio_shadow_state.json"),
            "latest_poll_summary": str(out_dir / "latest_portfolio_shadow_poll_summary.json"),
            "evidence_json": str(evidence_json),
            "evidence_md": str(evidence_md),
            "cycle_summary": str(cycle_summary_path),
            "cycle_md": str(cycle_md),
        },
    }
    paper_review_json = _resolve(args.paper_review_json)
    paper_review_md = _resolve(args.paper_review_md)
    if args.run_paper_review_gate:
        cycle_summary_path.parent.mkdir(parents=True, exist_ok=True)
        cycle_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        paper_review = evaluate_paper_review_gate(
            _paper_review_args(
                active_registry=_resolve(args.active_registry),
                manifest=manifest_path,
                validation_json=_resolve(args.validation_json) if args.validation_json else None,
                cycle_summary=cycle_summary_path,
                evidence_json=evidence_json,
            )
        )
        paper_review_json.parent.mkdir(parents=True, exist_ok=True)
        paper_review_json.write_text(json.dumps(paper_review, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        _write_paper_review_markdown(paper_review_md, paper_review)
        summary["paper_review_gate"] = {
            "decision": paper_review["decision"]["status"],
            "ready_for_manual_paper_review": paper_review["ready_for_manual_paper_review"],
            "ready_for_automated_paper": paper_review["ready_for_automated_paper"],
            "ready_for_paper": paper_review["ready_for_paper"],
            "paper_trading_automation": paper_review["paper_trading_automation"],
            "failed_gates": paper_review["decision"]["failed_gates"],
            "out_json": str(paper_review_json),
            "out_md": str(paper_review_md),
        }
        summary["outputs"]["paper_review_json"] = str(paper_review_json)
        summary["outputs"]["paper_review_md"] = str(paper_review_md)
    cycle_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_cycle_markdown(cycle_md, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase53 portfolio no-order shadow cycle and evidence gate.")
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("wfa_optimized_params_output/live_shadow_phase53_portfolio"))
    parser.add_argument("--cycle-summary", type=Path, default=Path("wfa_optimized_params_output/latest_portfolio_shadow_cycle_summary.json"))
    parser.add_argument("--evidence-json", type=Path, default=Path("wfa_optimized_params_output/portfolio_shadow_evidence_gate.json"))
    parser.add_argument("--evidence-md", type=Path, default=Path("wfa_optimized_params_output/portfolio_shadow_evidence_gate.md"))
    parser.add_argument("--run-paper-review-gate", action="store_true")
    parser.add_argument("--active-registry", type=Path, default=Path("wfa_optimized_params_output/phase53_active_portfolio_shadow_registry.json"))
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--paper-review-json", type=Path, default=Path("wfa_optimized_params_output/portfolio_paper_review_gate.json"))
    parser.add_argument("--paper-review-md", type=Path, default=Path("wfa_optimized_params_output/portfolio_paper_review_gate.md"))
    parser.add_argument("--component-id", action="append", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--signal-start", default=None, help="Replay exact signal timestamps from this UTC time.")
    parser.add_argument("--signal-end", default=None, help="Replay exact signal timestamps through this UTC time.")
    parser.add_argument("--max-polls", type=int, default=0)
    parser.add_argument("--update-data", action="store_true")
    parser.add_argument("--update-funding", action="store_true")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--overlap-bars", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--funding-start", default=None)
    parser.add_argument("--funding-end", default=None)
    parser.add_argument("--funding-overlap-events", type=int, default=3)
    parser.add_argument("--funding-limit", type=int, default=1000)
    parser.add_argument("--funding-timeout", type=float, default=30.0)
    parser.add_argument("--funding-sleep-seconds", type=float, default=0.1)
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
    parser.add_argument("--max-funding-lag-hours", type=float, default=12.0)
    parser.add_argument("--allow-stale-funding", action="store_true")
    args = parser.parse_args()

    summary = run_cycle(args)
    print(
        json.dumps(
            {
                "observation_mode": summary["observation_mode"],
                "poll_count": summary["poll_count"],
                "new_events": summary["event_totals"]["new"],
                "duplicates": summary["event_totals"]["duplicates"],
                "accepted_components": summary["event_totals"]["accepted_components"],
                "funding_coverage": summary["funding_coverage"],
                "evidence_decision": summary["evidence_gate"]["decision"],
                "paper_review_decision": summary["paper_review_gate"].get("decision", summary["paper_review_gate"].get("reason")),
                "effective_ready_for_manual_paper_review": summary["effective_ready_for_manual_paper_review"],
                "ready_for_paper": summary["ready_for_paper"],
                "cycle_summary": summary["outputs"]["cycle_summary"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
