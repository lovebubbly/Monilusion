from __future__ import annotations

import argparse
import csv
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
from run_live_shadow_poll import validate_shadow_manifest


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


def _select_cuda_row(cuda_obj: dict[str, Any], rank: int) -> dict[str, Any]:
    for row in cuda_obj.get("results", []):
        if int(row.get("rank", -1)) == rank:
            return row
    raise SystemExit(f"No CUDA result row for rank {rank}.")


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True).tz_convert(None)


def _filter_trades(
    trades: list[dict[str, Any]],
    *,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    if since is None and until is None:
        return trades
    out = []
    for trade in trades:
        ts = _parse_timestamp(trade.get("entry_signal_time"))
        if ts is None:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        out.append(trade)
    return out


def _event_from_trade(
    *,
    trade: dict[str, Any],
    candidate: dict[str, Any],
    candidate_params: dict[str, Any],
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "shadow_replay",
        "execution_permission": "NO_ORDERS_SHADOW_LOGGING_ONLY",
        "rank": int(candidate["rank"]),
        "param_id": candidate["param_id"],
        "signal_time": trade.get("entry_signal_time"),
        "entry_time": trade.get("entry_time"),
        "exit_time": trade.get("exit_time"),
        "side": trade.get("side"),
        "theoretical_entry_price": trade.get("entry_price"),
        "theoretical_exit_price": trade.get("exit_price"),
        "initial_stop_loss": trade.get("initial_stop_loss"),
        "take_profit": trade.get("take_profit"),
        "position_size": trade.get("position_size"),
        "exit_reason": trade.get("exit_reason"),
        "bars_held": trade.get("bars_held"),
        "gross_pnl": trade.get("gross_pnl"),
        "net_pnl": trade.get("net_pnl"),
        "funding": trade.get("funding"),
        "balance_after": trade.get("balance_after"),
        "entry_features": {
            key: trade.get(key)
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
        "risk_settings": {
            "risk_per_trade_percentage": candidate_params.get("risk_per_trade_percentage"),
            "atr_multiplier_sl": candidate_params.get("atr_multiplier_sl"),
            "risk_reward_ratio": candidate_params.get("risk_reward_ratio"),
            "entry_delay_bars": assumptions.get("entry_delay_bars"),
            "commission_rate": assumptions.get("commission_rate"),
            "slippage_rate": assumptions.get("slippage_rate"),
            "funding_rate_per_8h": assumptions.get("funding_rate_per_8h"),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "rank",
        "param_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "side",
        "theoretical_entry_price",
        "theoretical_exit_price",
        "initial_stop_loss",
        "take_profit",
        "position_size",
        "exit_reason",
        "bars_held",
        "gross_pnl",
        "net_pnl",
        "funding",
        "balance_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_shadow_replay(
    *,
    manifest_path: Path,
    csv_path: Path,
    out_dir: Path,
    ranks: set[int] | None,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    validate_shadow_manifest(manifest)

    cuda_path = Path(manifest["source_cuda_results"])
    if not cuda_path.is_absolute():
        cuda_path = manifest_path.parent.parent / cuda_path if str(cuda_path).startswith("wfa_") else Path.cwd() / cuda_path
    cuda_obj = _load_json(cuda_path)
    assumptions = manifest["assumptions"]
    df = _load_ohlcv(csv_path, cuda_obj["period_start"], cuda_obj["period_end"])

    out_dir.mkdir(parents=True, exist_ok=True)
    all_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for candidate in manifest.get("candidates", []):
        rank = int(candidate["rank"])
        if ranks is not None and rank not in ranks:
            continue
        row = _select_cuda_row(cuda_obj, rank)
        params = row["parameters"]
        metrics, trades = cpu_reference_backtest(
            df,
            params,
            initial_balance=float(row["performance"].get("initial_balance", 10000.0)),
            commission_rate=float(assumptions["commission_rate"]),
            slippage_rate=float(assumptions["slippage_rate"]),
            entry_delay_bars=int(assumptions["entry_delay_bars"]),
            funding_rate_per_8h=float(assumptions["funding_rate_per_8h"]),
            include_trades=True,
        )
        filtered = _filter_trades(trades, since=since, until=until)
        events = [
            _event_from_trade(
                trade=trade,
                candidate=candidate,
                candidate_params=params,
                assumptions=assumptions,
            )
            for trade in filtered
        ]
        all_events.extend(events)
        summaries.append(
            {
                "rank": rank,
                "param_id": candidate["param_id"],
                "decision": candidate.get("decision", {}),
                "cpu_reference": metrics,
                "window_logged_events": len(events),
                "full_sample_trades": len(trades),
                "recent_events": events[-5:],
            }
        )

    all_events.sort(key=lambda row: (str(row.get("signal_time") or ""), int(row["rank"])))
    events_path = out_dir / "shadow_events.jsonl"
    csv_path_out = out_dir / "shadow_events.csv"
    summary_path = out_dir / "shadow_replay_summary.json"
    _write_jsonl(events_path, all_events)
    _write_csv(csv_path_out, all_events)

    summary = {
        "schema_version": 1,
        "mode": "shadow_replay",
        "execution_permission": "NO_ORDERS_SHADOW_LOGGING_ONLY",
        "source_manifest": str(manifest_path),
        "source_cuda_results": str(cuda_path),
        "source_csv": str(csv_path),
        "since": since.isoformat() if since is not None else None,
        "until": until.isoformat() if until is not None else None,
        "candidate_count": len(summaries),
        "event_count": len(all_events),
        "outputs": {
            "events_jsonl": str(events_path),
            "events_csv": str(csv_path_out),
        },
        "candidates": [
            {
                "rank": row["rank"],
                "param_id": row["param_id"],
                "decision": row["decision"],
                "cpu_reference": row["cpu_reference"],
                "window_logged_events": row["window_logged_events"],
                "full_sample_trades": row["full_sample_trades"],
                "recent_events": row["recent_events"],
            }
            for row in summaries
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay promoted candidates as no-order shadow logs.")
    parser.add_argument(
        "--shadow-manifest",
        type=Path,
        default=Path("wfa_optimized_params_output/phase14_shadow_candidates_2019_2025.json"),
    )
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wfa_optimized_params_output/shadow_replay_phase14_2019_2025"),
    )
    parser.add_argument("--rank", type=int, action="append", default=None, help="Candidate rank to replay.")
    parser.add_argument("--since", default=None, help="Optional entry-signal start timestamp.")
    parser.add_argument("--until", default=None, help="Optional entry-signal end timestamp.")
    args = parser.parse_args()

    manifest_path = args.shadow_manifest if args.shadow_manifest.is_absolute() else Path.cwd() / args.shadow_manifest
    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    summary = run_shadow_replay(
        manifest_path=manifest_path,
        csv_path=csv_path,
        out_dir=out_dir,
        ranks=set(args.rank) if args.rank else None,
        since=_parse_timestamp(args.since),
        until=_parse_timestamp(args.until),
    )
    print(json.dumps({
        "execution_permission": summary["execution_permission"],
        "candidate_count": summary["candidate_count"],
        "event_count": summary["event_count"],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
