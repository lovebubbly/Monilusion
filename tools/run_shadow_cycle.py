from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from evaluate_shadow_paper_gate import _write_markdown, evaluate_gate
from run_live_shadow_poll import run_live_shadow_poll
from update_binance_futures_ohlcv import _parse_time_ms, update_ohlcv


def _json_default(obj: Any) -> Any:
    return str(obj)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def run_cycle(
    *,
    csv_path: Path,
    shadow_manifest: Path,
    live_shadow_dir: Path,
    paper_gate_json: Path,
    paper_gate_md: Path,
    cycle_summary: Path,
    extended_oos: Path | None,
    update_data: bool,
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
    overlap_bars: int,
    rank: set[int] | None,
    min_shadow_days: float,
    min_unique_signal_times: int,
    min_accepted_signals: int,
    require_modern_validation: bool,
) -> dict[str, Any]:
    data_update = None
    if update_data:
        data_update = update_ohlcv(
            csv_path=csv_path,
            symbol=symbol,
            interval=interval,
            start_ms=_parse_time_ms(start),
            end_ms=_parse_time_ms(end),
            overlap_bars=overlap_bars,
            limit=1000,
            timeout=30.0,
            sleep_seconds=0.1,
            write=True,
            backup=True,
        )
    live_summary = run_live_shadow_poll(
        manifest_path=shadow_manifest,
        csv_path=csv_path,
        out_dir=live_shadow_dir,
        ranks=rank,
        signal_time=None,
        as_of=None,
    )
    gate = evaluate_gate(
        manifest_path=shadow_manifest,
        events_path=live_shadow_dir / "live_shadow_events.jsonl",
        min_shadow_days=min_shadow_days,
        min_unique_signal_times=min_unique_signal_times,
        min_accepted_signals=min_accepted_signals,
        extended_oos_path=extended_oos,
        require_modern_validation=require_modern_validation,
    )
    paper_gate_json.parent.mkdir(parents=True, exist_ok=True)
    paper_gate_json.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(paper_gate_md, gate)
    summary = {
        "schema_version": 1,
        "mode": "shadow_cycle",
        "data_update": data_update or {"skipped": True, "reason": "update_data_false"},
        "live_shadow_poll": live_summary,
        "paper_gate": {
            "decision": gate["decision"],
            "ready_for_manual_paper_review": gate["ready_for_manual_paper_review"],
            "ready_for_automated_paper": gate["ready_for_automated_paper"],
            "failure_reasons": gate["failure_reasons"],
            "out_json": str(paper_gate_json),
            "out_md": str(paper_gate_md),
        },
    }
    cycle_summary.parent.mkdir(parents=True, exist_ok=True)
    cycle_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the no-order shadow cycle: optional data update, live poll, paper gate.")
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument(
        "--shadow-manifest",
        type=Path,
        default=Path("wfa_optimized_params_output/phase14_shadow_candidates_2019_2025.json"),
    )
    parser.add_argument("--live-shadow-dir", type=Path, default=Path("wfa_optimized_params_output/live_shadow_phase14"))
    parser.add_argument("--paper-gate-json", type=Path, default=Path("wfa_optimized_params_output/paper_gate_phase14_live_shadow.json"))
    parser.add_argument("--paper-gate-md", type=Path, default=Path("wfa_optimized_params_output/paper_gate_phase14_live_shadow.md"))
    parser.add_argument("--cycle-summary", type=Path, default=Path("wfa_optimized_params_output/latest_shadow_cycle_summary.json"))
    parser.add_argument("--extended-oos", type=Path, default=None)
    parser.add_argument("--update-data", action="store_true", help="Fetch Binance USD-M klines and write them to --csv.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--overlap-bars", type=int, default=6)
    parser.add_argument("--rank", type=int, action="append", default=None)
    parser.add_argument("--min-shadow-days", type=float, default=14.0)
    parser.add_argument("--min-unique-signal-times", type=int, default=20)
    parser.add_argument("--min-accepted-signals", type=int, default=3)
    parser.add_argument("--allow-legacy-manifest", action="store_true", help="Do not require schema v2 intrabar-band and purged-CPCV evidence for paper review.")
    args = parser.parse_args()

    summary = run_cycle(
        csv_path=_resolve(args.csv),
        shadow_manifest=_resolve(args.shadow_manifest),
        live_shadow_dir=_resolve(args.live_shadow_dir),
        paper_gate_json=_resolve(args.paper_gate_json),
        paper_gate_md=_resolve(args.paper_gate_md),
        cycle_summary=_resolve(args.cycle_summary),
        extended_oos=_resolve(args.extended_oos) if args.extended_oos is not None else None,
        update_data=args.update_data,
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        overlap_bars=args.overlap_bars,
        rank=set(args.rank) if args.rank else None,
        min_shadow_days=args.min_shadow_days,
        min_unique_signal_times=args.min_unique_signal_times,
        min_accepted_signals=args.min_accepted_signals,
        require_modern_validation=not args.allow_legacy_manifest,
    )
    print(json.dumps({
        "data_update": summary["data_update"],
        "live_shadow_new_events": summary["live_shadow_poll"]["new_event_count"],
        "live_shadow_duplicates": summary["live_shadow_poll"]["duplicate_event_count"],
        "paper_gate_decision": summary["paper_gate"]["decision"],
        "ready_for_automated_paper": summary["paper_gate"]["ready_for_automated_paper"],
        "cycle_summary": str(_resolve(args.cycle_summary)),
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
