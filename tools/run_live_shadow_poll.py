from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest


NO_ORDER_PERMISSION = "NO_ORDERS_SHADOW_LOGGING_ONLY"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_shadow_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("decision") != "PROMOTE_TO_SHADOW":
        raise SystemExit(
            "Refusing to run live shadow polling because manifest decision "
            f"is {manifest.get('decision')!r}."
        )
    if manifest.get("execution_permission") != NO_ORDER_PERMISSION:
        raise SystemExit(
            "Refusing to run live shadow polling because manifest execution_permission "
            f"is {manifest.get('execution_permission')!r}."
        )
    if manifest.get("ready_for_shadow") is not True:
        raise SystemExit("Refusing to poll because manifest is not ready_for_shadow.")
    if manifest.get("ready_for_paper") is True:
        raise SystemExit("Refusing to poll because manifest already marks ready_for_paper.")
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise SystemExit("Refusing to poll because manifest has no promoted candidates.")
    for candidate in candidates:
        decision = candidate.get("decision", {})
        if decision.get("status") != "PROMOTE_TO_SHADOW" or decision.get("ready_for_shadow") is not True:
            raise SystemExit(
                "Refusing to poll because candidate "
                f"rank {candidate.get('rank')} is not PROMOTE_TO_SHADOW."
            )
        if decision.get("failed_gates"):
            raise SystemExit(
                "Refusing to poll because candidate "
                f"rank {candidate.get('rank')} still has failed gates: {decision.get('failed_gates')}."
            )
        diff_pass = candidate.get("source_period_diff_pass", candidate.get("source_period_cpu_gpu_diff_pass", True))
        if diff_pass is not True:
            raise SystemExit(
                "Refusing to poll because candidate "
                f"rank {candidate.get('rank')} did not pass source CPU/GPU diff."
            )


def _resolve(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def _select_cuda_row(cuda_obj: dict[str, Any], rank: int) -> dict[str, Any]:
    for row in cuda_obj.get("results", []):
        if int(row.get("rank", -1)) == rank:
            return row
    raise SystemExit(f"No CUDA result row for rank {rank}.")


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True).tz_convert(None)


def _target_signal_time(
    df: pd.DataFrame,
    *,
    signal_time: pd.Timestamp | None,
    as_of: pd.Timestamp | None,
    entry_delay_bars: int,
) -> pd.Timestamp:
    if signal_time is not None:
        if signal_time not in df.index:
            raise SystemExit(f"signal_time {signal_time.isoformat()} is not present in the OHLCV index.")
        return signal_time
    visible = df if as_of is None else df[df.index <= as_of]
    if visible.empty:
        raise SystemExit("No OHLCV rows are visible at the requested as-of timestamp.")
    target_idx = len(visible) - 1 - max(0, entry_delay_bars)
    if target_idx < 1:
        raise SystemExit("Not enough closed bars to evaluate a next-bar-open signal.")
    return visible.index[target_idx]


def _event_id(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("rank")),
        str(row.get("signal_time")),
        str(row.get("status")),
        str(row.get("side") or row.get("raw_side")),
        ",".join(row.get("skip_reasons") or []),
        str(row.get("param_id")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "seen_event_ids": [], "last_signal_time_by_rank": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "event_id",
        "poll_time",
        "rank",
        "param_id",
        "signal_time",
        "status",
        "side",
        "raw_side",
        "planned_entry_time",
        "theoretical_entry_price",
        "initial_stop_loss",
        "take_profit",
        "position_size",
        "skip_reasons",
        "execution_permission",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            out["skip_reasons"] = ",".join(out.get("skip_reasons") or [])
            writer.writerow(out)


def _risk_settings(params: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_per_trade_percentage": params.get("risk_per_trade_percentage"),
        "atr_multiplier_sl": params.get("atr_multiplier_sl"),
        "risk_reward_ratio": params.get("risk_reward_ratio"),
        "entry_delay_bars": assumptions.get("entry_delay_bars"),
        "commission_rate": assumptions.get("commission_rate"),
        "slippage_rate": assumptions.get("slippage_rate"),
        "funding_rate_per_8h": assumptions.get("funding_rate_per_8h"),
    }


def _normalize_entry_event(
    *,
    event: dict[str, Any],
    rank: int,
    param_id: str,
    params: dict[str, Any],
    assumptions: dict[str, Any],
    poll_time: str,
) -> dict[str, Any]:
    status = event.get("status")
    out = {
        "mode": "live_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "poll_time": poll_time,
        "rank": rank,
        "param_id": param_id,
        "signal_time": event.get("entry_signal_time"),
        "status": status,
        "side": event.get("side"),
        "raw_side": event.get("raw_side"),
        "planned_entry_time": event.get("planned_entry_time") or event.get("entry_time"),
        "theoretical_entry_price": event.get("theoretical_entry_price"),
        "initial_stop_loss": event.get("initial_stop_loss"),
        "take_profit": event.get("take_profit"),
        "position_size": event.get("position_size"),
        "skip_reasons": event.get("skip_reasons") or [],
        "entry_features": {
            key: event.get(key)
            for key in [
                "entry_hour_utc",
                "entry_day_of_week",
                "entry_close",
                "entry_adx",
                "entry_rsi",
                "entry_atr_pct",
                "entry_ema_spread_atr",
                "entry_h1_slope_pct",
                "entry_h4_slope_pct",
            ]
        },
        "risk_settings": _risk_settings(params, assumptions),
    }
    out["event_id"] = _event_id(out)
    return out


def _no_signal_event(
    *,
    rank: int,
    param_id: str,
    target_time: pd.Timestamp,
    params: dict[str, Any],
    assumptions: dict[str, Any],
    poll_time: str,
) -> dict[str, Any]:
    out = {
        "mode": "live_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "poll_time": poll_time,
        "rank": rank,
        "param_id": param_id,
        "signal_time": target_time.isoformat(),
        "status": "no_entry_event",
        "side": None,
        "raw_side": None,
        "planned_entry_time": None,
        "theoretical_entry_price": None,
        "initial_stop_loss": None,
        "take_profit": None,
        "position_size": None,
        "skip_reasons": ["no_base_signal_or_pre_signal_filter_or_state_block"],
        "entry_features": {},
        "risk_settings": _risk_settings(params, assumptions),
    }
    out["event_id"] = _event_id(out)
    return out


def run_live_shadow_poll(
    *,
    manifest_path: Path,
    csv_path: Path,
    out_dir: Path,
    ranks: set[int] | None,
    signal_time: pd.Timestamp | None,
    as_of: pd.Timestamp | None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    validate_shadow_manifest(manifest)

    cuda_path = _resolve(Path(manifest["source_cuda_results"]), Path.cwd())
    cuda_obj = _load_json(cuda_path)
    assumptions = manifest["assumptions"]
    entry_delay_bars = int(assumptions.get("entry_delay_bars", 1))
    end = as_of.isoformat() if as_of is not None else "2100-01-01"
    df = _load_ohlcv(csv_path, cuda_obj["period_start"], end)
    target_time = _target_signal_time(
        df,
        signal_time=signal_time,
        as_of=as_of,
        entry_delay_bars=entry_delay_bars,
    )
    as_of_time = (as_of or df.index.max()).isoformat()
    poll_time = pd.Timestamp.utcnow().isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "live_shadow_state.json"
    jsonl_path = out_dir / "live_shadow_events.jsonl"
    csv_out_path = out_dir / "live_shadow_events.csv"
    summary_path = out_dir / "latest_live_shadow_poll_summary.json"
    state = _load_state(state_path)
    seen = set(state.get("seen_event_ids", []))

    new_events: list[dict[str, Any]] = []
    duplicate_events: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    for candidate in manifest.get("candidates", []):
        rank = int(candidate["rank"])
        if ranks is not None and rank not in ranks:
            continue
        row = _select_cuda_row(cuda_obj, rank)
        params = row["parameters"]
        _, _, entry_events = cpu_reference_backtest(
            df,
            params,
            initial_balance=float(row["performance"].get("initial_balance", 10000.0)),
            commission_rate=float(assumptions["commission_rate"]),
            slippage_rate=float(assumptions["slippage_rate"]),
            entry_delay_bars=entry_delay_bars,
            funding_rate_per_8h=float(assumptions["funding_rate_per_8h"]),
            include_trades=True,
            include_entry_events=True,
        )
        matching = [
            event
            for event in entry_events
            if pd.to_datetime(event.get("entry_signal_time"), utc=True).tz_convert(None) == target_time
        ]
        normalized = [
            _normalize_entry_event(
                event=event,
                rank=rank,
                param_id=candidate["param_id"],
                params=params,
                assumptions=assumptions,
                poll_time=poll_time,
            )
            for event in matching
        ] or [
            _no_signal_event(
                rank=rank,
                param_id=candidate["param_id"],
                target_time=target_time,
                params=params,
                assumptions=assumptions,
                poll_time=poll_time,
            )
        ]
        for event in normalized:
            if event["event_id"] in seen:
                duplicate_events.append(event)
            else:
                new_events.append(event)
                seen.add(event["event_id"])
        candidate_summaries.append(
            {
                "rank": rank,
                "param_id": candidate["param_id"],
                "events": [
                    {
                        "event_id": event["event_id"],
                        "status": event["status"],
                        "side": event.get("side"),
                        "raw_side": event.get("raw_side"),
                        "skip_reasons": event.get("skip_reasons") or [],
                    }
                    for event in normalized
                ],
            }
        )

    _append_jsonl(jsonl_path, new_events)
    _append_csv(csv_out_path, new_events)
    state["seen_event_ids"] = sorted(seen)
    state.setdefault("last_signal_time_by_rank", {})
    for event in new_events:
        state["last_signal_time_by_rank"][str(event["rank"])] = event["signal_time"]
    _save_state(state_path, state)

    summary = {
        "schema_version": 1,
        "mode": "live_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_paper": False,
        "source_manifest": str(manifest_path),
        "source_cuda_results": str(cuda_path),
        "source_csv": str(csv_path),
        "as_of": as_of_time,
        "target_signal_time": target_time.isoformat(),
        "candidate_count": len(candidate_summaries),
        "new_event_count": len(new_events),
        "duplicate_event_count": len(duplicate_events),
        "outputs": {
            "events_jsonl": str(jsonl_path),
            "events_csv": str(csv_out_path),
            "state": str(state_path),
        },
        "candidates": candidate_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one closed-candle no-order live shadow decision.")
    parser.add_argument(
        "--shadow-manifest",
        type=Path,
        default=Path("wfa_optimized_params_output/phase14_shadow_candidates_2019_2025.json"),
    )
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wfa_optimized_params_output/live_shadow_phase14"),
    )
    parser.add_argument("--rank", type=int, action="append", default=None)
    parser.add_argument("--signal-time", default=None, help="Exact signal candle timestamp to log.")
    parser.add_argument("--as-of", default=None, help="Only use candles at or before this timestamp.")
    args = parser.parse_args()

    manifest_path = _resolve(args.shadow_manifest, Path.cwd())
    csv_path = _resolve(args.csv, Path.cwd())
    out_dir = _resolve(args.out_dir, Path.cwd())
    summary = run_live_shadow_poll(
        manifest_path=manifest_path,
        csv_path=csv_path,
        out_dir=out_dir,
        ranks=set(args.rank) if args.rank else None,
        signal_time=_parse_ts(args.signal_time),
        as_of=_parse_ts(args.as_of),
    )
    print(json.dumps({
        "execution_permission": summary["execution_permission"],
        "target_signal_time": summary["target_signal_time"],
        "candidate_count": summary["candidate_count"],
        "new_event_count": summary["new_event_count"],
        "duplicate_event_count": summary["duplicate_event_count"],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
