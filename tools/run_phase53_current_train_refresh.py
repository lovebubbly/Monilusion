from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.backtest_v2 import load_csv  # noqa: E402
from tools.update_binance_futures_ohlcv import _parse_time_ms, update_ohlcv  # noqa: E402
from tools.update_binance_funding_rate import update_funding_rates  # noqa: E402
from tools.run_live_portfolio_shadow_poll import manifest_fingerprint  # noqa: E402


NO_ORDER_PERMISSION = "NO_ORDERS_SHADOW_LOGGING_ONLY"


@dataclass(frozen=True)
class Phase53Source:
    profile: str
    rank_metric: str
    top_k: int


PHASE53_SOURCES = [
    Phase53Source("phase45_donchian_htf_atr_cap", "robust", 100),
    Phase53Source("phase46_donchian_dd_taper", "robust", 100),
    Phase53Source("phase38_donchian_weekday_veto", "return", 100),
    Phase53Source("phase29_risk_scaled_breakout_selector", "return", 100),
]


def _resolve(path: str | Path | None, base: Path = ROOT) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _timestamp_for_cli(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S") if ts.tzinfo else ts.strftime("%Y-%m-%d %H:%M:%S")


def _stamp(ts: pd.Timestamp | None = None) -> str:
    if ts is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return ts.tz_convert("UTC").strftime("%Y%m%d_%H%M%S") if ts.tzinfo else ts.strftime("%Y%m%d_%H%M%S")


def _latest_data_timestamp(csv_path: Path) -> pd.Timestamp:
    df = load_csv(str(csv_path))
    if df.empty:
        raise SystemExit(f"CSV has no rows: {csv_path}")
    latest = pd.to_datetime(df["timestamp"], utc=True).max()
    if pd.isna(latest):
        raise SystemExit(f"CSV has no parseable timestamps: {csv_path}")
    return latest


def _result_glob(profile: str) -> str:
    return f"top_results_BTCUSDT_{profile}_*.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_new_cuda_result(profile: str, started_at: float, train_start: str, train_end: str) -> Path:
    candidates = []
    for path in (ROOT / "wfa_optimized_params_output").glob(_result_glob(profile)):
        if path.stat().st_mtime < started_at - 1.0:
            continue
        try:
            obj = _load_json(path)
        except json.JSONDecodeError:
            continue
        if obj.get("search_profile") != profile:
            continue
        if str(obj.get("period_start")) != train_start:
            continue
        if str(obj.get("period_end")) != train_end:
            continue
        candidates.append(path)
    if not candidates:
        raise SystemExit(f"No new CUDA result found for {profile} covering {train_start} to {train_end}.")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _run(cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "command": cmd,
        "started_at": datetime.fromtimestamp(started).isoformat(),
        "dry_run": dry_run,
    }
    if dry_run:
        row.update({"returncode": None, "elapsed_seconds": 0.0})
        return row
    proc = subprocess.run(cmd, cwd=ROOT)
    row.update({"returncode": proc.returncode, "elapsed_seconds": round(time.time() - started, 3)})
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return row


def _cuda_search_command(args: argparse.Namespace, source: Phase53Source, train_end: str) -> list[str]:
    cmd = [
        str(_resolve(args.python)),
        str(ROOT / "tools" / "run_cuda_strategy_search.py"),
        "--profile",
        source.profile,
        "--csv",
        str(_resolve(args.csv)),
        "--start",
        args.train_start,
        "--end",
        train_end,
        "--top-k",
        str(source.top_k),
        "--batch-size",
        str(args.batch_size),
        "--commission",
        str(args.commission),
        "--slippage",
        str(args.slippage),
        "--entry-delay-bars",
        str(args.entry_delay_bars),
        "--funding-rate-per-8h",
        str(args.funding_rate_per_8h),
        "--strict-min-return",
        str(args.strict_min_return),
        "--strict-min-pf",
        str(args.strict_min_pf),
        "--strict-max-mdd",
        str(args.strict_max_mdd),
        "--strict-min-trades",
        str(args.strict_min_trades),
        "--rank-metric",
        source.rank_metric,
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--python",
        str(_resolve(args.python)),
    ]
    funding_csv = _resolve(args.funding_csv)
    if funding_csv is not None:
        cmd.extend(["--funding-csv", str(funding_csv)])
    return cmd


def _export_command(args: argparse.Namespace, cuda_results: list[Path], manifest_out: Path) -> list[str]:
    cmd = [
        str(_resolve(args.python)),
        str(ROOT / "tools" / "export_portfolio_shadow_manifest.py"),
        "--portfolio-diagnostic",
        str(_resolve(args.portfolio_diagnostic)),
        "--selection-basis",
        "current_cuda_result",
    ]
    for result in cuda_results:
        cmd.extend(["--current-cuda-result", str(result)])
    cmd.extend(["--out", str(manifest_out)])
    return cmd


def _validate_command(args: argparse.Namespace, manifest: Path, train_end: str, out_json: Path, out_md: Path) -> list[str]:
    cmd = [
        str(_resolve(args.python)),
        str(ROOT / "tools" / "validate_portfolio_shadow_manifest.py"),
        "--portfolio-manifest",
        str(manifest),
        "--csv",
        str(_resolve(args.csv)),
        "--start",
        args.train_start,
        "--end",
        train_end,
        "--n-trials",
        str(args.n_trials),
        "--max-funding-lag-hours",
        str(args.max_funding_lag_hours),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if args.allow_stale_funding:
        cmd.append("--allow-stale-funding")
    return cmd


def _poll_command(args: argparse.Namespace, manifest: Path, out_dir: Path, as_of: str) -> list[str]:
    cmd = [
        str(_resolve(args.python)),
        str(ROOT / "tools" / "run_live_portfolio_shadow_poll.py"),
        "--portfolio-manifest",
        str(manifest),
        "--csv",
        str(_resolve(args.csv)),
        "--out-dir",
        str(out_dir),
        "--as-of",
        as_of,
        "--max-funding-lag-hours",
        str(args.max_funding_lag_hours),
    ]
    if args.allow_stale_funding:
        cmd.append("--allow-stale-funding")
    return cmd


def _evidence_command(args: argparse.Namespace, manifest: Path, events_jsonl: Path, out_json: Path, out_md: Path) -> list[str]:
    return [
        str(_resolve(args.python)),
        str(ROOT / "tools" / "evaluate_portfolio_shadow_evidence.py"),
        "--portfolio-manifest",
        str(manifest),
        "--events-jsonl",
        str(events_jsonl),
        "--min-observed-days",
        str(args.min_shadow_observed_days),
        "--min-unique-signal-times",
        str(args.min_shadow_unique_signal_times),
        "--min-total-events",
        str(args.min_shadow_total_events),
        "--min-accepted-events",
        str(args.min_shadow_accepted_events),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase53 Current Train Refresh",
        "",
        f"- Decision: `{summary['decision']['status']}`",
        f"- Execution permission: `{summary['execution_permission']}`",
        f"- Ready for paper: `false`",
        f"- Latest data timestamp: `{summary['refresh']['latest_data_timestamp']}`",
        f"- Train window: `{summary['refresh']['train_start']}` to `{summary['refresh']['train_end']}`",
        f"- Train-end embargo bars: `{summary['refresh']['train_end_embargo_bars']}`",
        f"- Data update: `{summary.get('data_update', {}).get('mode', summary.get('data_update', {}).get('reason'))}`",
        f"- Funding update: `{summary.get('funding_update', {}).get('mode', summary.get('funding_update', {}).get('reason'))}`",
        f"- Active manifest: `{summary.get('active_manifest', {}).get('path')}`",
        "",
        "| source | rank metric | cuda result |",
        "| --- | --- | --- |",
    ]
    for source in summary["sources"]:
        lines.append(f"| {source['profile']} | {source['rank_metric']} | `{source.get('cuda_result') or ''}` |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        if value:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_refresh(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = _resolve(args.csv)
    data_update: dict[str, Any] = {"skipped": True, "reason": "update_data_false"}
    if args.update_data:
        if args.dry_run:
            data_update = {
                "skipped": True,
                "reason": "dry_run",
                "csv_path": str(csv_path),
                "symbol": args.symbol,
                "interval": args.interval,
            }
        else:
            data_update = update_ohlcv(
                csv_path=csv_path,
                symbol=args.symbol,
                interval=args.interval,
                start_ms=_parse_time_ms(args.data_start),
                end_ms=_parse_time_ms(args.data_end),
                overlap_bars=args.overlap_bars,
                limit=args.data_limit,
                timeout=args.data_timeout,
                sleep_seconds=args.data_sleep_seconds,
                write=True,
                backup=True,
            )
    latest = _latest_data_timestamp(csv_path)
    train_end_ts = pd.to_datetime(args.train_end, utc=True) if args.train_end else latest - pd.Timedelta(hours=args.train_end_embargo_bars)
    if train_end_ts >= latest:
        raise SystemExit("Train end must be earlier than the latest visible data timestamp.")
    train_end = _timestamp_for_cli(train_end_ts)
    run_stamp = args.run_id or _stamp(latest)
    out_dir = _resolve(args.out_dir) / f"phase53_current_train_refresh_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    funding_update: dict[str, Any] = {"skipped": True, "reason": "update_funding_false"}
    if args.update_funding:
        funding_path = _resolve(args.funding_csv)
        if args.dry_run:
            funding_update = {
                "skipped": True,
                "reason": "dry_run",
                "csv_path": str(funding_path),
                "symbol": args.symbol,
            }
        else:
            funding_update = update_funding_rates(
                csv_path=funding_path,
                symbol=args.symbol,
                start_ms=None,
                end_ms=None,
                overlap_events=args.funding_overlap_events,
                limit=args.funding_limit,
                timeout=args.funding_timeout,
                sleep_seconds=args.funding_sleep_seconds,
                write=True,
                backup=True,
            )

    supplied_cuda = [_resolve(path) for path in args.current_cuda_result]
    if supplied_cuda and len(supplied_cuda) != len(PHASE53_SOURCES):
        raise SystemExit("--current-cuda-result must be supplied exactly four times when used.")
    if args.skip_search and len(supplied_cuda) != len(PHASE53_SOURCES):
        raise SystemExit("--skip-search requires four --current-cuda-result paths.")

    source_rows: list[dict[str, Any]] = []
    cuda_results: list[Path] = []
    command_rows: list[dict[str, Any]] = []

    if supplied_cuda:
        cuda_results = [Path(path) for path in supplied_cuda]
        for source, result in zip(PHASE53_SOURCES, cuda_results):
            obj = _load_json(result)
            if obj.get("search_profile") != source.profile:
                raise SystemExit(f"CUDA profile mismatch: {result} is not {source.profile}.")
            if str(obj.get("period_start")) != args.train_start or str(obj.get("period_end")) != train_end:
                raise SystemExit(
                    "Supplied CUDA result is not from the requested train-only window: "
                    f"{result} covers {obj.get('period_start')} to {obj.get('period_end')}, "
                    f"expected {args.train_start} to {train_end}."
                )
            source_rows.append(
                {
                    "profile": source.profile,
                    "rank_metric": source.rank_metric,
                    "top_k": source.top_k,
                    "cuda_result": str(result),
                    "search_skipped": True,
                }
            )
    else:
        for source in PHASE53_SOURCES:
            started_at = time.time()
            cmd = _cuda_search_command(args, source, train_end)
            row = _run(cmd, dry_run=args.dry_run)
            command_rows.append(row)
            result_path = None
            if not args.dry_run:
                result_path = _find_new_cuda_result(source.profile, started_at, args.train_start, train_end)
                cuda_results.append(result_path)
            source_rows.append(
                {
                    "profile": source.profile,
                    "rank_metric": source.rank_metric,
                    "top_k": source.top_k,
                    "cuda_result": str(result_path) if result_path else None,
                    "search_skipped": False,
                    "command": cmd,
                    "returncode": row.get("returncode"),
                }
            )

    manifest_out = out_dir / f"phase53_portfolio_shadow_manifest_current_train_{run_stamp}.json"
    validation_json = out_dir / f"validation_phase53_portfolio_shadow_current_train_{run_stamp}.json"
    validation_md = out_dir / f"validation_phase53_portfolio_shadow_current_train_{run_stamp}.md"
    smoke_dir = out_dir / "live_shadow_smoke"
    evidence_json = out_dir / f"shadow_evidence_gate_{run_stamp}.json"
    evidence_md = out_dir / f"shadow_evidence_gate_{run_stamp}.md"
    summary_json = out_dir / f"phase53_current_train_refresh_{run_stamp}.json"
    summary_md = out_dir / f"phase53_current_train_refresh_{run_stamp}.md"
    active_manifest_out = _resolve(args.active_manifest_out)
    active_registry_out = _resolve(args.active_registry_out)

    if cuda_results:
        export_row = _run(_export_command(args, cuda_results, manifest_out), dry_run=args.dry_run)
        command_rows.append(export_row)
        validate_row = _run(_validate_command(args, manifest_out, train_end, validation_json, validation_md), dry_run=args.dry_run)
        command_rows.append(validate_row)
        if args.run_shadow_smoke:
            as_of = args.smoke_as_of or _timestamp_for_cli(latest)
            poll_row = _run(_poll_command(args, manifest_out, smoke_dir, as_of), dry_run=args.dry_run)
            command_rows.append(poll_row)
            if args.run_evidence_gate:
                events_jsonl = smoke_dir / "portfolio_shadow_events.jsonl"
                evidence_row = _run(
                    _evidence_command(args, manifest_out, events_jsonl, evidence_json, evidence_md),
                    dry_run=args.dry_run,
                )
                command_rows.append(evidence_row)

    validation_decision = None
    failed_gates: list[str] = []
    ready_for_shadow = False
    if validation_json.exists():
        validation = _load_json(validation_json)
        validation_decision = validation.get("decision", {})
        failed_gates = list(validation_decision.get("failed_gates") or [])
        ready_for_shadow = validation_decision.get("ready_for_shadow") is True and not failed_gates

    status = "DRY_RUN_ONLY" if args.dry_run else ("CURRENT_TRAIN_REFRESH_READY_FOR_SHADOW" if ready_for_shadow else "HOLD_AUTOMATED_PAPER")
    manifest_hash = manifest_fingerprint(manifest_out) if manifest_out.exists() else None
    active_manifest: dict[str, Any] = {
        "updated": False,
        "path": str(active_manifest_out),
        "registry": str(active_registry_out),
        "manifest_sha256": manifest_hash,
        "reason": "not_ready_for_shadow" if not ready_for_shadow else "ready",
    }
    if ready_for_shadow and manifest_out.exists() and active_manifest_out is not None and active_registry_out is not None:
        active_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        active_registry_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_out, active_manifest_out)
        active_manifest_hash = manifest_fingerprint(active_manifest_out)
        registry = {
            "schema_version": 1,
            "mode": "phase53_active_portfolio_shadow_manifest",
            "updated_at": datetime.now().isoformat(),
            "execution_permission": NO_ORDER_PERMISSION,
            "ready_for_paper": False,
            "paper_trading_automation": "HOLD",
            "active_manifest": str(active_manifest_out),
            "active_manifest_sha256": active_manifest_hash,
            "source_manifest": str(manifest_out),
            "source_manifest_sha256": manifest_hash,
            "validation_json": str(validation_json) if validation_json.exists() else None,
            "refresh_summary": str(summary_json),
            "train_start": args.train_start,
            "train_end": train_end,
            "latest_data_timestamp": latest.isoformat(),
            "decision": status,
        }
        active_registry_out.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        active_manifest.update(
            {
                "updated": True,
                "manifest_sha256": active_manifest_hash,
                "source_manifest": str(manifest_out),
                "source_manifest_sha256": manifest_hash,
                "reason": "ready_for_shadow",
            }
        )
    summary = {
        "schema_version": 1,
        "mode": "phase53_current_train_refresh",
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "refresh": {
            "latest_data_timestamp": latest.isoformat(),
            "train_start": args.train_start,
            "train_end": train_end,
            "train_end_embargo_bars": args.train_end_embargo_bars,
            "dry_run": args.dry_run,
            "run_id": run_stamp,
        },
        "data_update": data_update,
        "funding_update": funding_update,
        "manifest_sha256": manifest_hash,
        "active_manifest": active_manifest,
        "sources": source_rows,
        "commands": command_rows,
        "outputs": {
            "out_dir": str(out_dir),
            "manifest": str(manifest_out) if manifest_out.exists() or args.dry_run else None,
            "active_manifest": str(active_manifest_out),
            "active_registry": str(active_registry_out),
            "validation_json": str(validation_json) if validation_json.exists() or args.dry_run else None,
            "validation_md": str(validation_md) if validation_md.exists() or args.dry_run else None,
            "shadow_smoke_dir": str(smoke_dir) if smoke_dir.exists() or args.dry_run else None,
            "shadow_evidence_json": str(evidence_json) if evidence_json.exists() or args.dry_run else None,
            "shadow_evidence_md": str(evidence_md) if evidence_md.exists() or args.dry_run else None,
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
        "decision": {
            "status": status,
            "ready_for_shadow": ready_for_shadow,
            "ready_for_paper": False,
            "failed_gates": failed_gates,
            "validation_decision": validation_decision,
            "rationale": (
                "Current train-only refresh passed validation for no-order shadow observation."
                if ready_for_shadow
                else "Automated paper remains held until refresh validation and live shadow evidence pass."
            ),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(summary_md, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase53 current train-only CUDA refresh, export no-order shadow manifest, validate, and smoke poll."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument("--portfolio-diagnostic", type=Path, default=Path("wfa_optimized_params_output/wfo_portfolio_phase53_dsr_meta2_full_n1771_20260517.json"))
    parser.add_argument("--funding-csv", type=Path, default=Path("wfa_optimized_params_output/futures_context/BTCUSDT_funding_rate_8h_20190101_20260516.csv"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--train-start", default="2019-12-24")
    parser.add_argument("--train-end", default=None, help="Explicit UTC train end timestamp. Defaults to latest data minus embargo bars.")
    parser.add_argument("--train-end-embargo-bars", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=Path("wfa_optimized_params_output"))
    parser.add_argument("--active-manifest-out", type=Path, default=Path("wfa_optimized_params_output/phase53_active_portfolio_shadow_manifest.json"))
    parser.add_argument("--active-registry-out", type=Path, default=Path("wfa_optimized_params_output/phase53_active_portfolio_shadow_registry.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0001)
    parser.add_argument("--strict-min-return", type=float, default=30.0)
    parser.add_argument("--strict-min-pf", type=float, default=1.3)
    parser.add_argument("--strict-max-mdd", type=float, default=25.0)
    parser.add_argument("--strict-min-trades", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=1771)
    parser.add_argument("--current-cuda-result", action="append", type=Path, default=[])
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--update-data", action="store_true")
    parser.add_argument("--data-start", default=None)
    parser.add_argument("--data-end", default=None)
    parser.add_argument("--overlap-bars", type=int, default=6)
    parser.add_argument("--data-limit", type=int, default=1000)
    parser.add_argument("--data-timeout", type=float, default=30.0)
    parser.add_argument("--data-sleep-seconds", type=float, default=0.1)
    parser.add_argument("--update-funding", action="store_true")
    parser.add_argument("--funding-overlap-events", type=int, default=3)
    parser.add_argument("--funding-limit", type=int, default=1000)
    parser.add_argument("--funding-timeout", type=float, default=30.0)
    parser.add_argument("--funding-sleep-seconds", type=float, default=0.1)
    parser.add_argument("--run-shadow-smoke", action="store_true")
    parser.add_argument("--smoke-as-of", default=None)
    parser.add_argument("--run-evidence-gate", action="store_true")
    parser.add_argument("--min-shadow-observed-days", type=float, default=14.0)
    parser.add_argument("--min-shadow-unique-signal-times", type=int, default=24)
    parser.add_argument("--min-shadow-total-events", type=int, default=84)
    parser.add_argument("--min-shadow-accepted-events", type=int, default=1)
    parser.add_argument("--max-funding-lag-hours", type=float, default=12.0)
    parser.add_argument("--allow-stale-funding", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_refresh(args)
    print(
        json.dumps(
            {
                "decision": summary["decision"]["status"],
                "ready_for_shadow": summary["decision"]["ready_for_shadow"],
                "ready_for_paper": summary["decision"]["ready_for_paper"],
                "train_end": summary["refresh"]["train_end"],
                "sources": [
                    {"profile": row["profile"], "cuda_result": row.get("cuda_result")}
                    for row in summary["sources"]
                ],
                "outputs": summary["outputs"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
