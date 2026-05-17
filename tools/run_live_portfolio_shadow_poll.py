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

from diff_cuda_cpu_reference import _load_ohlcv, cpu_reference_backtest, load_funding_cumulative


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


def manifest_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True).tz_convert(None)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _funding_event_times(path: Path) -> pd.Series:
    if not path.exists():
        raise SystemExit(f"Funding CSV not found: {path}")
    frame = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in frame.columns}
    time_col = cols.get("funding_time") or cols.get("timestamp") or cols.get("time")
    rate_col = cols.get("funding_rate") or cols.get("fundingrate")
    if time_col is None or rate_col is None:
        raise SystemExit(f"Funding CSV must contain funding_time/timestamp and funding_rate columns: {path}")
    numeric = pd.to_numeric(frame[time_col], errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(frame) * 0.9)):
        unit = "ms" if numeric.dropna().median() > 10_000_000_000 else "s"
        times = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        times = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    times = times.dropna().dt.tz_convert(None).sort_values()
    if times.empty:
        raise SystemExit(f"Funding CSV has no valid funding timestamps: {path}")
    return times


def _validate_funding_coverage(
    *,
    assumptions: dict[str, Any],
    df: pd.DataFrame,
    max_lag_hours: float,
    allow_stale: bool,
) -> dict[str, Any]:
    funding_model = assumptions.get("funding_model")
    funding_csv = assumptions.get("funding_rate_csv")
    if funding_model != "actual_funding_events":
        return {
            "enabled": False,
            "funding_model": funding_model,
            "status": "not_required",
        }
    if not funding_csv:
        raise SystemExit("Manifest uses actual funding events but has no funding_rate_csv.")
    funding_path = _resolve(Path(funding_csv), Path.cwd())
    times = _funding_event_times(funding_path)
    visible_start = df.index.min()
    visible_end = df.index.max()
    first_event = times.iloc[0]
    last_event = times.iloc[-1]
    lag_hours = (visible_end - last_event).total_seconds() / 3600.0
    starts_before_data = first_event <= visible_start
    lag_ok = lag_hours <= max_lag_hours
    status = "ok" if starts_before_data and lag_ok else "stale_or_incomplete"
    coverage = {
        "enabled": True,
        "funding_model": funding_model,
        "funding_rate_csv": str(funding_path),
        "first_funding_time": first_event.isoformat(),
        "last_funding_time": last_event.isoformat(),
        "visible_start": visible_start.isoformat(),
        "visible_end": visible_end.isoformat(),
        "lag_hours": round(lag_hours, 6),
        "max_lag_hours": max_lag_hours,
        "starts_before_visible_data": bool(starts_before_data),
        "status": status,
        "allow_stale": bool(allow_stale),
    }
    if status != "ok" and not allow_stale:
        raise SystemExit(f"Funding coverage is stale or incomplete: {coverage}")
    return coverage


def validate_portfolio_shadow_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_type") != "portfolio_shadow":
        raise SystemExit("Refusing to poll because manifest_type is not portfolio_shadow.")
    if int(manifest.get("schema_version", 0)) < 3:
        raise SystemExit("Refusing to poll because portfolio manifest schema_version is below 3.")
    if manifest.get("decision") != "PROMOTE_TO_SHADOW":
        raise SystemExit(f"Refusing to poll because manifest decision is {manifest.get('decision')!r}.")
    if manifest.get("execution_permission") != NO_ORDER_PERMISSION:
        raise SystemExit("Refusing to poll because execution_permission is not no-order shadow logging.")
    if manifest.get("ready_for_shadow") is not True:
        raise SystemExit("Refusing to poll because manifest is not ready_for_shadow.")
    if manifest.get("ready_for_paper") is True or manifest.get("paper_trading_automation") != "HOLD":
        raise SystemExit("Refusing to poll because manifest does not keep paper trading disabled.")
    assumptions = manifest.get("assumptions", {})
    if not isinstance(assumptions, dict):
        raise SystemExit("Refusing to poll because manifest has no execution assumptions.")
    if assumptions.get("market") != "Binance USD-M perpetual":
        raise SystemExit("Refusing to poll because manifest market is not Binance USD-M perpetual.")
    if int(assumptions.get("entry_delay_bars", 0)) != 1:
        raise SystemExit("Refusing to poll because entry_delay_bars is not 1 next-bar-open.")
    if _num(assumptions.get("commission_rate"), 0.0) < 0.0005:
        raise SystemExit("Refusing to poll because commission_rate is below the conservative fee assumption.")
    if _num(assumptions.get("slippage_rate"), 0.0) < 0.0002:
        raise SystemExit("Refusing to poll because slippage_rate is below the conservative slippage assumption.")
    if assumptions.get("funding_model") != "actual_funding_events" or not assumptions.get("funding_rate_csv"):
        raise SystemExit("Refusing to poll because actual funding events are not configured.")
    if assumptions.get("intrabar_policy_base") != "conservative":
        raise SystemExit("Refusing to poll because intrabar_policy_base is not conservative.")
    if assumptions.get("shadow_execution") != "no_orders_log_signals_only":
        raise SystemExit("Refusing to poll because shadow_execution is not no-orders logging.")
    if assumptions.get("paper_trading") != "disabled":
        raise SystemExit("Refusing to poll because paper_trading is not disabled in assumptions.")
    components = manifest.get("components", [])
    if not components:
        raise SystemExit("Refusing to poll because manifest has no portfolio components.")
    total_weight = sum(_num(row.get("component_weight")) for row in components)
    if abs(total_weight - 1.0) > 1e-6:
        raise SystemExit(f"Refusing to poll because component weights sum to {total_weight}, not 1.0.")
    for component in components:
        if not isinstance(component.get("parameters"), dict):
            raise SystemExit(f"Component {component.get('component_id')} is missing parameters.")
        decision = component.get("decision", {})
        if decision.get("ready_for_shadow") is not True or decision.get("ready_for_paper") is True:
            raise SystemExit(f"Component {component.get('component_id')} is not safe for no-order shadow.")
        if decision.get("failed_gates"):
            raise SystemExit(f"Component {component.get('component_id')} still has failed gates.")


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
        str(row.get("manifest_sha256")),
        str(row.get("portfolio_label")),
        str(row.get("component_id")),
        str(row.get("signal_time")),
        str(row.get("status")),
        str(row.get("side") or row.get("raw_side")),
        ",".join(row.get("skip_reasons") or []),
        str(row.get("param_id")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "seen_event_ids": [], "last_signal_time_by_component": {}}
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
        "manifest_sha256",
        "source_manifest",
        "observation_mode",
        "live_evidence_eligible",
        "poll_time",
        "portfolio_label",
        "component_id",
        "component_weight",
        "source_profile",
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
        "portfolio_position_size",
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


def _risk_settings(params: dict[str, Any], assumptions: dict[str, Any], component: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_weight": component.get("component_weight"),
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
    component: dict[str, Any],
    manifest: dict[str, Any],
    assumptions: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    poll_time: str,
    observation_mode: str,
    live_evidence_eligible: bool,
) -> dict[str, Any]:
    params = component["parameters"]
    component_weight = _num(component.get("component_weight"))
    position_size = event.get("position_size")
    out = {
        "mode": "live_portfolio_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "source_manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "observation_mode": observation_mode,
        "live_evidence_eligible": bool(live_evidence_eligible),
        "poll_time": poll_time,
        "portfolio_label": manifest.get("portfolio", {}).get("label"),
        "component_id": component["component_id"],
        "component_weight": component_weight,
        "source_profile": component.get("source_profile"),
        "param_id": component.get("param_id"),
        "signal_time": event.get("entry_signal_time"),
        "status": event.get("status"),
        "side": event.get("side"),
        "raw_side": event.get("raw_side"),
        "planned_entry_time": event.get("planned_entry_time") or event.get("entry_time"),
        "theoretical_entry_price": event.get("theoretical_entry_price"),
        "initial_stop_loss": event.get("initial_stop_loss"),
        "take_profit": event.get("take_profit"),
        "position_size": position_size,
        "portfolio_position_size": position_size,
        "skip_reasons": event.get("skip_reasons") or [],
        "entry_features": {
            key: event.get(key)
            for key in [
                "entry_hour_utc",
                "entry_day_of_week",
                "entry_close",
                "entry_adx",
                "entry_h4_adx",
                "entry_rsi",
                "entry_atr_pct",
                "entry_h4_atr_pct",
                "entry_ema_spread_atr",
                "entry_h1_slope_pct",
                "entry_h4_slope_pct",
            ]
        },
        "risk_settings": _risk_settings(params, assumptions, component),
    }
    out["event_id"] = _event_id(out)
    return out


def _no_signal_event(
    *,
    target_time: pd.Timestamp,
    component: dict[str, Any],
    manifest: dict[str, Any],
    assumptions: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    poll_time: str,
    observation_mode: str,
    live_evidence_eligible: bool,
) -> dict[str, Any]:
    params = component["parameters"]
    out = {
        "mode": "live_portfolio_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "source_manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "observation_mode": observation_mode,
        "live_evidence_eligible": bool(live_evidence_eligible),
        "poll_time": poll_time,
        "portfolio_label": manifest.get("portfolio", {}).get("label"),
        "component_id": component["component_id"],
        "component_weight": component.get("component_weight"),
        "source_profile": component.get("source_profile"),
        "param_id": component.get("param_id"),
        "signal_time": target_time.isoformat(),
        "status": "no_entry_event",
        "side": None,
        "raw_side": None,
        "planned_entry_time": None,
        "theoretical_entry_price": None,
        "initial_stop_loss": None,
        "take_profit": None,
        "position_size": None,
        "portfolio_position_size": None,
        "skip_reasons": ["no_base_signal_or_pre_signal_filter_or_state_block"],
        "entry_features": {},
        "risk_settings": _risk_settings(params, assumptions, component),
    }
    out["event_id"] = _event_id(out)
    return out


def run_live_portfolio_shadow_poll(
    *,
    manifest_path: Path,
    csv_path: Path,
    out_dir: Path,
    component_ids: set[str] | None,
    signal_time: pd.Timestamp | None,
    as_of: pd.Timestamp | None,
    max_funding_lag_hours: float = 12.0,
    allow_stale_funding: bool = False,
    observation_mode: str = "latest_live_closed_candle",
    live_evidence_eligible: bool = True,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    manifest_sha256 = manifest_fingerprint(manifest_path)
    validate_portfolio_shadow_manifest(manifest)
    assumptions = manifest.get("assumptions", {})
    entry_delay_bars = int(assumptions.get("entry_delay_bars", 1))
    end = as_of.isoformat() if as_of is not None else "2100-01-01"
    df = _load_ohlcv(csv_path, "2019-12-24", end)
    target_time = _target_signal_time(
        df,
        signal_time=signal_time,
        as_of=as_of,
        entry_delay_bars=entry_delay_bars,
    )
    funding_coverage = _validate_funding_coverage(
        assumptions=assumptions,
        df=df,
        max_lag_hours=max_funding_lag_hours,
        allow_stale=allow_stale_funding,
    )
    as_of_time = (as_of or df.index.max()).isoformat()
    poll_time = pd.Timestamp.utcnow().isoformat()
    funding_csv = assumptions.get("funding_rate_csv")
    funding_cumulative = None
    if funding_csv:
        funding_path = _resolve(Path(funding_csv), Path.cwd())
        if funding_path.exists():
            funding_cumulative = load_funding_cumulative(df, funding_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "portfolio_shadow_state.json"
    jsonl_path = out_dir / "portfolio_shadow_events.jsonl"
    csv_out_path = out_dir / "portfolio_shadow_events.csv"
    summary_path = out_dir / "latest_portfolio_shadow_poll_summary.json"
    state = _load_state(state_path)
    seen = set(state.get("seen_event_ids", []))

    new_events: list[dict[str, Any]] = []
    duplicate_events: list[dict[str, Any]] = []
    component_summaries: list[dict[str, Any]] = []
    initial = _num(assumptions.get("initial_balance"), 10_000.0)

    for component in manifest.get("components", []):
        component_id = str(component["component_id"])
        if component_ids is not None and component_id not in component_ids:
            continue
        component_weight = _num(component.get("component_weight"))
        component_initial = initial * component_weight
        _, _, entry_events = cpu_reference_backtest(
            df,
            component["parameters"],
            initial_balance=component_initial,
            commission_rate=_num(assumptions.get("commission_rate"), 0.0005),
            slippage_rate=_num(assumptions.get("slippage_rate"), 0.0002),
            entry_delay_bars=entry_delay_bars,
            funding_rate_per_8h=_num(assumptions.get("funding_rate_per_8h"), 0.0),
            include_trades=True,
            include_entry_events=True,
            funding_cumulative=funding_cumulative,
        )
        matching = [
            event
            for event in entry_events
            if pd.to_datetime(event.get("entry_signal_time"), utc=True).tz_convert(None) == target_time
        ]
        normalized = [
            _normalize_entry_event(
                event=event,
                component=component,
                manifest=manifest,
                assumptions=assumptions,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                poll_time=poll_time,
                observation_mode=observation_mode,
                live_evidence_eligible=live_evidence_eligible,
            )
            for event in matching
        ] or [
            _no_signal_event(
                target_time=target_time,
                component=component,
                manifest=manifest,
                assumptions=assumptions,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                poll_time=poll_time,
                observation_mode=observation_mode,
                live_evidence_eligible=live_evidence_eligible,
            )
        ]
        for event in normalized:
            if event["event_id"] in seen:
                duplicate_events.append(event)
            else:
                new_events.append(event)
                seen.add(event["event_id"])
        component_summaries.append(
            {
                "component_id": component_id,
                "component_weight": component_weight,
                "source_profile": component.get("source_profile"),
                "param_id": component.get("param_id"),
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
    state.setdefault("last_signal_time_by_component", {})
    for event in new_events:
        state["last_signal_time_by_component"][event["component_id"]] = event["signal_time"]
    _save_state(state_path, state)

    summary = {
        "schema_version": 1,
        "mode": "live_portfolio_shadow_poll",
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "source_manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "observation_mode": observation_mode,
        "live_evidence_eligible": bool(live_evidence_eligible),
        "source_csv": str(csv_path),
        "funding_coverage": funding_coverage,
        "as_of": as_of_time,
        "target_signal_time": target_time.isoformat(),
        "component_count": len(component_summaries),
        "new_event_count": len(new_events),
        "duplicate_event_count": len(duplicate_events),
        "accepted_component_count": sum(
            1 for row in component_summaries for event in row["events"] if event["status"] == "accepted"
        ),
        "outputs": {
            "events_jsonl": str(jsonl_path),
            "events_csv": str(csv_out_path),
            "state": str(state_path),
        },
        "components": component_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one closed-candle no-order live portfolio shadow decision.")
    parser.add_argument("--portfolio-manifest", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1h.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("wfa_optimized_params_output/live_shadow_phase53_portfolio"))
    parser.add_argument("--component-id", action="append", default=None)
    parser.add_argument("--signal-time", default=None, help="Exact signal candle timestamp to log.")
    parser.add_argument("--as-of", default=None, help="Only use candles at or before this timestamp.")
    parser.add_argument("--max-funding-lag-hours", type=float, default=12.0)
    parser.add_argument("--allow-stale-funding", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve(args.portfolio_manifest, Path.cwd())
    csv_path = _resolve(args.csv, Path.cwd())
    out_dir = _resolve(args.out_dir, Path.cwd())
    summary = run_live_portfolio_shadow_poll(
        manifest_path=manifest_path,
        csv_path=csv_path,
        out_dir=out_dir,
        component_ids=set(args.component_id) if args.component_id else None,
        signal_time=_parse_ts(args.signal_time),
        as_of=_parse_ts(args.as_of),
        max_funding_lag_hours=args.max_funding_lag_hours,
        allow_stale_funding=args.allow_stale_funding,
    )
    print(
        json.dumps(
            {
                "execution_permission": summary["execution_permission"],
                "ready_for_paper": summary["ready_for_paper"],
        "funding_coverage": summary["funding_coverage"],
        "manifest_sha256": summary["manifest_sha256"],
        "observation_mode": summary["observation_mode"],
        "live_evidence_eligible": summary["live_evidence_eligible"],
        "target_signal_time": summary["target_signal_time"],
                "component_count": summary["component_count"],
                "accepted_component_count": summary["accepted_component_count"],
                "new_event_count": summary["new_event_count"],
                "duplicate_event_count": summary["duplicate_event_count"],
                "outputs": summary["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
