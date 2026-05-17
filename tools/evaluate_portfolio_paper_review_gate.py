from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from run_live_portfolio_shadow_poll import NO_ORDER_PERMISSION, manifest_fingerprint, validate_portfolio_shadow_manifest  # noqa: E402


def _resolve(path: str | Path | None, base: Path = ROOT) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_path(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return _resolve(left).resolve() == _resolve(right).resolve()  # type: ignore[union-attr]
    except OSError:
        return str(_resolve(left)) == str(_resolve(right))


def _gate(name: str, passed: bool, observed: Any, threshold: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _component_signature(rows: list[dict[str, Any]], *, weight_key: str) -> list[dict[str, Any]]:
    signature = []
    for row in rows:
        signature.append(
            {
                "component_id": str(row.get("component_id")),
                "weight": round(_num(row.get(weight_key)), 10),
                "source_profile": row.get("source_profile"),
                "rank": row.get("rank"),
                "param_id": row.get("param_id"),
            }
        )
    return sorted(signature, key=lambda row: row["component_id"])


def _source_cuda_diff_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature = []
    for row in rows:
        source_cuda = row.get("source_cuda_results")
        signature.append(
            {
                "component_id": str(row.get("component_id")),
                "source_profile": row.get("source_profile"),
                "rank": row.get("rank"),
                "param_id": row.get("param_id"),
                "source_cuda_results": str(_resolve(source_cuda)) if source_cuda else None,
            }
        )
    return sorted(signature, key=lambda row: row["component_id"])


def _source_cuda_diff_rows_valid(rows: list[dict[str, Any]]) -> bool:
    required_diff_keys = {
        "total_net_pnl_percentage",
        "num_trades",
        "net_profit_factor",
        "gross_profit_factor",
        "max_drawdown_percentage",
    }
    if not rows:
        return False
    for row in rows:
        if row.get("diff_pass") is not True:
            return False
        diff = row.get("diff") or {}
        if not required_diff_keys.issubset(diff):
            return False
        intrabar = row.get("intrabar_policy_comparison") or {}
        if not intrabar.get("conservative") or not intrabar.get("optimistic"):
            return False
    return True


def _portfolio_meta_gates_pass(meta: dict[str, Any]) -> bool:
    gates = meta.get("gates") or []
    return bool(gates) and all(gate.get("pass") is True for gate in gates)


def _portfolio_diagnostic_snapshot_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    portfolio = manifest.get("portfolio", {})
    meta = portfolio.get("portfolio_meta_selection", {})
    meta_summary = meta.get("summary", {})
    meta_dsr = meta_summary.get("dsr", {})
    return {
        "source_portfolio_diagnostic": str(_resolve(manifest.get("source_portfolio_diagnostic")))
        if manifest.get("source_portfolio_diagnostic")
        else None,
        "component_count": int(portfolio.get("component_count", 0)),
        "source_count": int(portfolio.get("source_count", 0)),
        "fixed_return_pct": _num(portfolio.get("fixed_weight_summary", {}).get("return_pct")),
        "fixed_mdd_pct": _num(portfolio.get("fixed_weight_summary", {}).get("mdd_pct")),
        "fixed_trades": int(_num(portfolio.get("fixed_weight_summary", {}).get("trades"), 0.0)),
        "fixed_trade_profit_factor": _num(portfolio.get("fixed_weight_summary", {}).get("trade_profit_factor")),
        "fixed_dsr": _num(portfolio.get("fixed_weight_dsr", {}).get("dsr")),
        "fixed_mc_return_p05": _num(portfolio.get("monte_carlo", {}).get("return_pct_p05"), -1e9),
        "fixed_mc_prob_return_positive": _num(portfolio.get("monte_carlo", {}).get("prob_return_positive")),
        "cost_stress_pass_ratio": _num(portfolio.get("cost_stress_pass_ratio")),
        "pbo_enabled": portfolio.get("portfolio_weight_pbo", {}).get("enabled") is True,
        "pbo": _num(portfolio.get("portfolio_weight_pbo", {}).get("pbo"), 1.0),
        "pbo_samples": int(_num(portfolio.get("portfolio_weight_pbo", {}).get("samples"), 0.0)),
        "pbo_weight_candidates": int(_num(portfolio.get("portfolio_weight_pbo", {}).get("weight_candidates"), 0.0)),
        "pbo_score_metric": portfolio.get("portfolio_weight_pbo", {}).get("score_metric"),
        "pbo_median_test_percentile": _num(
            portfolio.get("portfolio_weight_pbo", {}).get("median_test_percentile"),
            0.0,
        ),
        "meta_enabled": meta.get("enabled") is True,
        "meta_return_pct": _num(meta_summary.get("return_pct")),
        "meta_mdd_pct": _num(meta_summary.get("mdd_pct")),
        "meta_dsr": _num(meta_dsr.get("dsr")),
        "meta_mc_return_p05": _num(meta_summary.get("monte_carlo", {}).get("return_pct_p05"), -1e9),
        "meta_gates_pass": _portfolio_meta_gates_pass(meta),
    }


def _portfolio_diagnostic_snapshot_from_source(source: dict[str, Any], source_path: Path | None) -> dict[str, Any]:
    portfolio = source.get("portfolio", {})
    meta = source.get("portfolio_meta_selection", {})
    meta_summary = meta.get("summary", {})
    meta_dsr = meta_summary.get("dsr", {})
    formal_gate = source.get("formal_promotion_gate", {})
    return {
        "source_portfolio_diagnostic": str(source_path) if source_path is not None else None,
        "component_count": len(source.get("sources", [])),
        "source_count": len(source.get("sources", [])),
        "fixed_return_pct": _num(portfolio.get("summary", {}).get("return_pct")),
        "fixed_mdd_pct": _num(portfolio.get("summary", {}).get("mdd_pct")),
        "fixed_trades": int(_num(portfolio.get("summary", {}).get("trades"), 0.0)),
        "fixed_trade_profit_factor": _num(portfolio.get("summary", {}).get("trade_profit_factor")),
        "fixed_dsr": _num(portfolio.get("dsr", {}).get("dsr")),
        "fixed_mc_return_p05": _num(portfolio.get("monte_carlo", {}).get("return_pct_p05"), -1e9),
        "fixed_mc_prob_return_positive": _num(portfolio.get("monte_carlo", {}).get("prob_return_positive")),
        "cost_stress_pass_ratio": _num(source.get("cost_stress", {}).get("pass_ratio")),
        "pbo_enabled": source.get("portfolio_weight_pbo", {}).get("enabled") is True,
        "pbo": _num(source.get("portfolio_weight_pbo", {}).get("pbo"), 1.0),
        "pbo_samples": int(_num(source.get("portfolio_weight_pbo", {}).get("samples"), 0.0)),
        "pbo_weight_candidates": int(_num(source.get("portfolio_weight_pbo", {}).get("weight_candidates"), 0.0)),
        "pbo_score_metric": source.get("portfolio_weight_pbo", {}).get("score_metric"),
        "pbo_median_test_percentile": _num(source.get("portfolio_weight_pbo", {}).get("median_test_percentile"), 0.0),
        "meta_enabled": meta.get("enabled") is True,
        "meta_return_pct": _num(meta_summary.get("return_pct")),
        "meta_mdd_pct": _num(meta_summary.get("mdd_pct")),
        "meta_dsr": _num(meta_dsr.get("dsr")),
        "meta_mc_return_p05": _num(meta_summary.get("monte_carlo", {}).get("return_pct_p05"), -1e9),
        "meta_gates_pass": _portfolio_meta_gates_pass(meta),
        "formal_failed_gates": sorted(formal_gate.get("failed_gates") or []),
        "formal_pbo_gate_pass": formal_gate.get("pbo_gate_pass") is True,
        "formal_meta_gate_pass": formal_gate.get("meta_gate_pass") is True,
        "formal_stress_gate_pass": formal_gate.get("stress_gate_pass") is True,
    }


def _portfolio_diagnostic_strict(snapshot: dict[str, Any]) -> bool:
    return (
        int(snapshot.get("component_count", 0)) > 0
        and int(snapshot.get("source_count", 0)) > 0
        and _num(snapshot.get("fixed_return_pct"), 0.0) >= 30.0
        and _num(snapshot.get("fixed_mdd_pct"), 1e9) <= 25.0
        and _num(snapshot.get("fixed_trade_profit_factor"), 0.0) >= 1.3
        and _num(snapshot.get("fixed_dsr"), 0.0) >= 0.95
        and _num(snapshot.get("fixed_mc_return_p05"), -1e9) >= 0.0
        and _num(snapshot.get("fixed_mc_prob_return_positive"), 0.0) >= 0.95
        and _num(snapshot.get("cost_stress_pass_ratio"), 0.0) >= 1.0
        and snapshot.get("pbo_enabled") is True
        and _num(snapshot.get("pbo"), 1.0) <= 0.5
        and int(snapshot.get("pbo_samples", 0)) > 0
        and int(snapshot.get("pbo_weight_candidates", 0)) >= 100
        and snapshot.get("pbo_score_metric") == "dsr"
        and _num(snapshot.get("pbo_median_test_percentile"), 0.0) >= 0.5
        and snapshot.get("meta_enabled") is True
        and _num(snapshot.get("meta_return_pct"), 0.0) >= 30.0
        and _num(snapshot.get("meta_mdd_pct"), 1e9) <= 25.0
        and _num(snapshot.get("meta_dsr"), 0.0) >= 0.95
        and _num(snapshot.get("meta_mc_return_p05"), -1e9) >= 0.0
        and snapshot.get("meta_gates_pass") is True
    )


def _source_portfolio_diagnostic_strict(snapshot: dict[str, Any]) -> bool:
    return (
        _portfolio_diagnostic_strict(snapshot)
        and snapshot.get("formal_failed_gates") == ["formal_portfolio_export"]
        and snapshot.get("formal_pbo_gate_pass") is True
        and snapshot.get("formal_meta_gate_pass") is True
        and snapshot.get("formal_stress_gate_pass") is True
    )


def _diagnostic_snapshots_match(manifest_snapshot: dict[str, Any], source_snapshot: dict[str, Any]) -> bool:
    keys = [
        "fixed_return_pct",
        "fixed_mdd_pct",
        "fixed_trades",
        "fixed_trade_profit_factor",
        "fixed_dsr",
        "fixed_mc_return_p05",
        "fixed_mc_prob_return_positive",
        "cost_stress_pass_ratio",
        "pbo_enabled",
        "pbo",
        "pbo_samples",
        "pbo_weight_candidates",
        "pbo_score_metric",
        "pbo_median_test_percentile",
        "meta_enabled",
        "meta_return_pct",
        "meta_mdd_pct",
        "meta_dsr",
        "meta_mc_return_p05",
        "meta_gates_pass",
    ]
    return all(manifest_snapshot.get(key) == source_snapshot.get(key) for key in keys)


def _execution_assumption_snapshot(assumptions: dict[str, Any]) -> dict[str, Any]:
    funding_csv = _resolve(assumptions.get("funding_rate_csv")) if assumptions.get("funding_rate_csv") else None
    return {
        "market": assumptions.get("market"),
        "commission_rate": _num(assumptions.get("commission_rate"), 0.0),
        "slippage_rate": _num(assumptions.get("slippage_rate"), 0.0),
        "funding_rate_per_8h": _num(assumptions.get("funding_rate_per_8h"), 0.0),
        "funding_model": assumptions.get("funding_model"),
        "funding_rate_csv": str(funding_csv) if funding_csv is not None else None,
        "entry_delay_bars": int(assumptions.get("entry_delay_bars", 0)),
        "entry_price_policy": "next_bar_open",
        "intrabar_policy_base": assumptions.get("intrabar_policy_base", "conservative"),
        "drawdown_basis": "intrabar_mark_to_market_equity_curve",
        "profit_factor_reporting": "component_gross_and_net_pf_plus_portfolio_net_trade_pf",
        "cpu_reference_engine": "tools.diff_cuda_cpu_reference.cpu_reference_backtest",
        "htf_alignment": "4h_label_right_closed_left_ffill_with_runtime_no_lookahead_check",
    }


def _strict_execution_assumptions(snapshot: dict[str, Any]) -> bool:
    return (
        snapshot.get("market") == "Binance USD-M perpetual"
        and int(snapshot.get("entry_delay_bars", 0)) == 1
        and _num(snapshot.get("commission_rate"), 0.0) >= 0.0005
        and _num(snapshot.get("slippage_rate"), 0.0) >= 0.0002
        and snapshot.get("funding_model") == "actual_funding_events"
        and bool(snapshot.get("funding_rate_csv"))
        and snapshot.get("intrabar_policy_base") == "conservative"
        and snapshot.get("entry_price_policy") == "next_bar_open"
        and snapshot.get("drawdown_basis") == "intrabar_mark_to_market_equity_curve"
        and snapshot.get("profit_factor_reporting") == "component_gross_and_net_pf_plus_portfolio_net_trade_pf"
        and snapshot.get("cpu_reference_engine") == "tools.diff_cuda_cpu_reference.cpu_reference_backtest"
        and snapshot.get("htf_alignment") == "4h_label_right_closed_left_ffill_with_runtime_no_lookahead_check"
    )


def _manifest_validation_gate(manifest: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate_portfolio_shadow_manifest(manifest)
    except SystemExit as exc:
        return False, str(exc)
    return True, None


def _format_observed_for_markdown(gate: dict[str, Any]) -> str:
    observed = gate.get("observed")
    if gate.get("name") == "validation_components_match_manifest" and isinstance(observed, dict):
        manifest_components = observed.get("manifest_components") or []
        validation_components = observed.get("validation_components") or []
        return {
            "manifest_component_count": len(manifest_components),
            "validation_component_count": len(validation_components),
            "match": manifest_components == validation_components,
        }.__repr__()
    text = repr(observed)
    if len(text) > 700:
        return text[:697] + "..."
    return text


def evaluate_paper_review_gate(args: argparse.Namespace) -> dict[str, Any]:
    registry_path = _resolve(args.active_registry)
    registry = _load_json(registry_path)
    active_manifest_path = _resolve(args.portfolio_manifest or registry.get("active_manifest"))
    evidence_path = _resolve(args.evidence_json)
    cycle_summary_path = _resolve(args.cycle_summary)
    validation_path = _resolve(args.validation_json or registry.get("validation_json"))

    manifest = _load_json(active_manifest_path)
    evidence = _load_json(evidence_path)
    cycle = _load_json(cycle_summary_path)
    validation = _load_json(validation_path) if validation_path and validation_path.exists() else {}

    manifest_sha256 = manifest_fingerprint(active_manifest_path)
    registry_hash = registry.get("active_manifest_sha256")
    validation_decision = validation.get("decision", {})
    validation_source_manifest_path = _resolve(validation.get("source_manifest")) if validation.get("source_manifest") else None
    validation_source_manifest_hash = (
        manifest_fingerprint(validation_source_manifest_path)
        if validation_source_manifest_path is not None and validation_source_manifest_path.exists()
        else None
    )
    manifest_component_signature = _component_signature(manifest.get("components", []), weight_key="component_weight")
    validation_component_signature = _component_signature(validation.get("components", []), weight_key="weight")
    manifest_execution_assumptions = _execution_assumption_snapshot(manifest.get("assumptions", {}))
    validation_execution_assumptions = validation.get("execution_assumptions", {})
    validation_source_cpu_cuda_diff = validation.get("source_cpu_cuda_diff", {})
    manifest_source_cuda_diff_signature = _source_cuda_diff_signature(manifest.get("components", []))
    validation_source_cuda_diff_rows = validation.get("component_source_cpu_cuda_diffs", [])
    validation_source_cuda_diff_signature = _source_cuda_diff_signature(validation_source_cuda_diff_rows)
    portfolio_diagnostic_path = _resolve(manifest.get("source_portfolio_diagnostic")) if manifest.get("source_portfolio_diagnostic") else None
    source_portfolio_diagnostic = (
        _load_json(portfolio_diagnostic_path) if portfolio_diagnostic_path is not None and portfolio_diagnostic_path.exists() else {}
    )
    manifest_portfolio_diagnostic = _portfolio_diagnostic_snapshot_from_manifest(manifest)
    source_portfolio_snapshot = _portfolio_diagnostic_snapshot_from_source(
        source_portfolio_diagnostic,
        portfolio_diagnostic_path if portfolio_diagnostic_path and portfolio_diagnostic_path.exists() else None,
    )
    cycle_poll = (cycle.get("polls") or [{}])[-1]
    funding_coverage = cycle.get("funding_coverage") or cycle_poll.get("funding_coverage") or {}
    manifest_valid, manifest_validation_error = _manifest_validation_gate(manifest)
    train_end = _parse_utc(registry.get("train_end"))
    cycle_target_signal_time = _parse_utc(cycle_poll.get("target_signal_time"))
    evidence_summary = evidence.get("summary", {})
    evidence_first_signal_time = _parse_utc(evidence_summary.get("first_signal_time"))
    evidence_last_signal_time = _parse_utc(evidence_summary.get("last_signal_time"))

    gates = [
        _gate(
            "registry_mode",
            registry.get("mode") == "phase53_active_portfolio_shadow_manifest",
            registry.get("mode"),
            "== phase53_active_portfolio_shadow_manifest",
            "Use the stable active Phase53 portfolio registry.",
        ),
        _gate(
            "registry_no_order_only",
            registry.get("execution_permission") == NO_ORDER_PERMISSION
            and registry.get("ready_for_paper") is not True
            and registry.get("paper_trading_automation") == "HOLD",
            {
                "execution_permission": registry.get("execution_permission"),
                "ready_for_paper": registry.get("ready_for_paper"),
                "paper_trading_automation": registry.get("paper_trading_automation"),
            },
            f"permission == {NO_ORDER_PERMISSION}, ready_for_paper != true, automation == HOLD",
            "Registry must never bypass the no-order shadow state.",
        ),
        _gate(
            "manifest_matches_registry_path",
            _same_path(active_manifest_path, registry.get("active_manifest")),
            {"manifest": str(active_manifest_path), "registry": registry.get("active_manifest")},
            "resolved paths match",
            "Paper review must evaluate the active registered manifest.",
        ),
        _gate(
            "manifest_matches_registry_hash",
            manifest_sha256 == registry_hash,
            {"manifest_sha256": manifest_sha256, "registry_sha256": registry_hash},
            "hashes match",
            "Active manifest contents must match the registry fingerprint.",
        ),
        _gate(
            "manifest_valid_for_shadow",
            manifest_valid,
            manifest_validation_error or "ok",
            "portfolio shadow manifest validator passes",
            "Manifest must still satisfy schema-v3 no-order portfolio constraints.",
        ),
        _gate(
            "validation_decision_ready_for_shadow",
            validation_decision.get("status") == "CURRENT_COMPONENTS_VALIDATED_FOR_SHADOW_OBSERVATION"
            and validation_decision.get("ready_for_shadow") is True
            and validation_decision.get("ready_for_paper") is not True
            and not validation_decision.get("failed_gates"),
            validation_decision,
            "current components validated for shadow, no failed gates, paper false",
            "CPU reference validation must still support the active component set.",
        ),
        _gate(
            "validation_source_manifest_matches_registry",
            _same_path(validation.get("source_manifest"), registry.get("source_manifest")),
            {"validation_source_manifest": validation.get("source_manifest"), "registry_source_manifest": registry.get("source_manifest")},
            "resolved paths match",
            "Validation JSON must point to the manifest source recorded by the active registry.",
        ),
        _gate(
            "validation_source_manifest_hash_matches_registry",
            validation_source_manifest_hash is not None
            and validation_source_manifest_hash == registry.get("source_manifest_sha256")
            and validation_source_manifest_hash == manifest_sha256,
            {
                "validation_source_manifest_sha256": validation_source_manifest_hash,
                "registry_source_manifest_sha256": registry.get("source_manifest_sha256"),
                "active_manifest_sha256": manifest_sha256,
            },
            "validation source hash == registry source hash == active manifest hash",
            "The validated source manifest must be byte-identical to the active manifest.",
        ),
        _gate(
            "validation_components_match_manifest",
            bool(manifest_component_signature)
            and manifest_component_signature == validation_component_signature,
            {
                "manifest_components": manifest_component_signature,
                "validation_components": validation_component_signature,
            },
            "component signatures match",
            "CPU validation must cover exactly the active manifest component ids, weights, ranks, profiles, and params.",
        ),
        _gate(
            "manifest_execution_assumptions_strict",
            _strict_execution_assumptions(manifest_execution_assumptions),
            manifest_execution_assumptions,
            "Binance USD-M, next-bar-open, conservative intrabar, actual funding, fee/slippage, MTM DD, gross/net PF",
            "Active manifest must encode the execution assumptions required for real-market validation.",
        ),
        _gate(
            "validation_execution_assumptions_match_manifest",
            _strict_execution_assumptions(validation_execution_assumptions)
            and validation_execution_assumptions == manifest_execution_assumptions,
            {
                "manifest": manifest_execution_assumptions,
                "validation": validation_execution_assumptions,
            },
            "validation execution assumptions exactly match manifest assumptions",
            "CPU validation evidence must be produced under the same execution assumptions as the active manifest.",
        ),
        _gate(
            "validation_source_cpu_cuda_diff_pass",
            validation_source_cpu_cuda_diff.get("all_pass") is True
            and validation_source_cpu_cuda_diff.get("component_count") == len(manifest_component_signature)
            and not validation_source_cpu_cuda_diff.get("failed_components"),
            validation_source_cpu_cuda_diff,
            "all active components CPU-vs-CUDA diff pass",
            "Validation evidence must prove each active component matches its CUDA/fast-search source row.",
        ),
        _gate(
            "validation_source_cpu_cuda_diff_details_match",
            manifest_source_cuda_diff_signature == validation_source_cuda_diff_signature
            and _source_cuda_diff_rows_valid(validation_source_cuda_diff_rows),
            {
                "manifest_components": manifest_source_cuda_diff_signature,
                "validation_diff_components": validation_source_cuda_diff_signature,
                "detail_rows_valid": _source_cuda_diff_rows_valid(validation_source_cuda_diff_rows),
            },
            "component source CUDA diff rows match active manifest and each row diff-passes",
            "CPU/CUDA diff detail rows must correspond exactly to active manifest components, ranks, params, and source CUDA files.",
        ),
        _gate(
            "manifest_portfolio_diagnostic_strict",
            _portfolio_diagnostic_strict(manifest_portfolio_diagnostic),
            manifest_portfolio_diagnostic,
            "fixed DSR/MC/stress/PBO/meta-selection portfolio diagnostic gates pass",
            "Active manifest must retain strict portfolio WFO/PBO/DSR/MC/cost-stress diagnostic evidence.",
        ),
        _gate(
            "source_portfolio_diagnostic_strict",
            bool(source_portfolio_diagnostic) and _source_portfolio_diagnostic_strict(source_portfolio_snapshot),
            source_portfolio_snapshot,
            "source diagnostic passes except the historical formal export placeholder gate",
            "Source Phase53 portfolio diagnostic must still prove PBO/meta-selection/stress gates before paper review.",
        ),
        _gate(
            "portfolio_diagnostic_source_matches_manifest",
            bool(source_portfolio_diagnostic)
            and _diagnostic_snapshots_match(manifest_portfolio_diagnostic, source_portfolio_snapshot),
            {
                "manifest": manifest_portfolio_diagnostic,
                "source": source_portfolio_snapshot,
            },
            "embedded manifest diagnostic snapshot matches source diagnostic",
            "Manifest-embedded portfolio diagnostic evidence must match the source diagnostic file.",
        ),
        _gate(
            "cycle_is_live_observation",
            cycle.get("observation_mode") == "latest_live_closed_candle",
            cycle.get("observation_mode"),
            "== latest_live_closed_candle",
            "Historical holdout replays cannot unlock paper review.",
        ),
        _gate(
            "cycle_manifest_hash_matches",
            cycle_poll.get("manifest_sha256") == manifest_sha256,
            {"cycle_manifest_sha256": cycle_poll.get("manifest_sha256"), "active_manifest_sha256": manifest_sha256},
            "hashes match",
            "Latest cycle must be tied to the active manifest.",
        ),
        _gate(
            "cycle_target_after_train_end",
            train_end is not None
            and cycle_target_signal_time is not None
            and cycle_target_signal_time > train_end,
            {
                "train_end": train_end.isoformat() if train_end else None,
                "cycle_target_signal_time": cycle_target_signal_time.isoformat() if cycle_target_signal_time else None,
            },
            "cycle target signal time > active registry train_end",
            "Live cycle used for paper review must be after the component-selection training window.",
        ),
        _gate(
            "funding_coverage_ok",
            funding_coverage.get("status") == "ok",
            funding_coverage,
            "status == ok",
            "Live cycle must use fresh enough actual funding data.",
        ),
        _gate(
            "evidence_manifest_hash_matches",
            evidence.get("manifest_sha256") == manifest_sha256,
            {"evidence_manifest_sha256": evidence.get("manifest_sha256"), "active_manifest_sha256": manifest_sha256},
            "hashes match",
            "Evidence gate must evaluate events from the active manifest.",
        ),
        _gate(
            "evidence_hash_integrity",
            evidence_summary.get("missing_manifest_hash_count") == 0
            and evidence_summary.get("mismatched_manifest_hash_count") == 0,
            {
                "missing": evidence_summary.get("missing_manifest_hash_count"),
                "mismatched": evidence_summary.get("mismatched_manifest_hash_count"),
            },
            "missing == 0 and mismatched == 0",
            "Every logged event must include and match the active manifest hash.",
        ),
        _gate(
            "evidence_after_train_end",
            train_end is not None
            and evidence_first_signal_time is not None
            and evidence_last_signal_time is not None
            and evidence_first_signal_time > train_end
            and evidence_last_signal_time > train_end,
            {
                "train_end": train_end.isoformat() if train_end else None,
                "first_signal_time": evidence_first_signal_time.isoformat() if evidence_first_signal_time else None,
                "last_signal_time": evidence_last_signal_time.isoformat() if evidence_last_signal_time else None,
            },
            "all evidence signal times > active registry train_end",
            "Paper-review evidence must not include in-sample or component-selection-window observations.",
        ),
        _gate(
            "evidence_live_only",
            evidence_summary.get("missing_observation_mode_count") == 0
            and evidence_summary.get("non_live_evidence_count") == 0,
            {
                "missing_observation_mode_count": evidence_summary.get("missing_observation_mode_count"),
                "non_live_evidence_count": evidence_summary.get("non_live_evidence_count"),
                "event_observation_modes": evidence_summary.get("event_observation_modes"),
            },
            "missing observation mode == 0 and non-live evidence == 0",
            "Paper-review evidence must be explicitly live-eligible and must not include historical replay events.",
        ),
        _gate(
            "shadow_evidence_ready",
            evidence.get("decision", {}).get("ready_for_manual_paper_review") is True,
            {
                "status": evidence.get("decision", {}).get("status"),
                "failed_gates": evidence.get("decision", {}).get("failed_gates"),
                "shortfalls": evidence.get("readiness", {}).get("shortfalls"),
            },
            "ready_for_manual_paper_review == true",
            "Enough live shadow evidence must exist before manual paper review.",
        ),
        _gate(
            "paper_automation_disabled",
            manifest.get("ready_for_paper") is not True
            and evidence.get("ready_for_paper") is not True
            and cycle.get("ready_for_paper") is not True
            and manifest.get("paper_trading_automation") == "HOLD"
            and evidence.get("paper_trading_automation") == "HOLD"
            and cycle.get("paper_trading_automation") == "HOLD",
            {
                "manifest_ready_for_paper": manifest.get("ready_for_paper"),
                "evidence_ready_for_paper": evidence.get("ready_for_paper"),
                "cycle_ready_for_paper": cycle.get("ready_for_paper"),
                "manifest_automation": manifest.get("paper_trading_automation"),
                "evidence_automation": evidence.get("paper_trading_automation"),
                "cycle_automation": cycle.get("paper_trading_automation"),
            },
            "ready_for_paper != true and automation == HOLD everywhere",
            "This gate may permit manual review, never automated paper trading.",
        ),
    ]
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    ready_for_manual_review = not failed
    return {
        "schema_version": 1,
        "mode": "phase53_portfolio_paper_review_gate",
        "active_registry": str(registry_path),
        "active_manifest": str(active_manifest_path),
        "active_manifest_sha256": manifest_sha256,
        "validation_json": str(validation_path) if validation_path else None,
        "validation_source_manifest": str(validation_source_manifest_path) if validation_source_manifest_path else None,
        "validation_source_manifest_sha256": validation_source_manifest_hash,
        "component_signature": {
            "active_manifest": manifest_component_signature,
            "validation": validation_component_signature,
        },
        "execution_assumptions": {
            "active_manifest": manifest_execution_assumptions,
            "validation": validation_execution_assumptions,
        },
        "portfolio_diagnostic": {
            "active_manifest": manifest_portfolio_diagnostic,
            "source": source_portfolio_snapshot,
        },
        "source_cpu_cuda_diff_signature": {
            "active_manifest": manifest_source_cuda_diff_signature,
            "validation": validation_source_cuda_diff_signature,
        },
        "cycle_summary": str(cycle_summary_path),
        "evidence_json": str(evidence_path),
        "train_end": train_end.isoformat() if train_end else None,
        "cycle_target_signal_time": cycle_target_signal_time.isoformat() if cycle_target_signal_time else None,
        "evidence_signal_window": {
            "first_signal_time": evidence_first_signal_time.isoformat() if evidence_first_signal_time else None,
            "last_signal_time": evidence_last_signal_time.isoformat() if evidence_last_signal_time else None,
        },
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_manual_paper_review": ready_for_manual_review,
        "ready_for_automated_paper": False,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "decision": {
            "status": "READY_FOR_MANUAL_PAPER_REVIEW" if ready_for_manual_review else "HOLD_PAPER_REVIEW",
            "failed_gates": failed,
            "rationale": (
                "All portfolio paper-review gates passed; automated paper remains disabled pending manual review."
                if ready_for_manual_review
                else "Portfolio is not ready for manual paper review; keep no-order shadow observation only."
            ),
        },
        "evidence_readiness": evidence.get("readiness", {}),
        "funding_coverage": funding_coverage,
        "gates": gates,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Phase53 Portfolio Paper Review Gate",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Ready for manual paper review: `{str(result['ready_for_manual_paper_review']).lower()}`",
        f"- Ready for automated paper: `false`",
        f"- Paper trading automation: `{result['paper_trading_automation']}`",
        f"- Active manifest SHA-256: `{result['active_manifest_sha256']}`",
        "",
        "| gate | pass | observed | threshold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in result["gates"]:
        lines.append(f"| {gate['name']} | {str(gate['pass']).lower()} | `{_format_observed_for_markdown(gate)}` | `{gate['threshold']}` |")
    lines.extend(["", "## Remaining Evidence", ""])
    shortfalls = result.get("evidence_readiness", {}).get("shortfalls", {})
    if shortfalls:
        for name in ("observed_days", "unique_signal_times", "total_events", "accepted_events"):
            item = shortfalls.get(name, {})
            lines.append(
                f"- {name}: observed `{item.get('observed')}`, threshold `{item.get('threshold')}`, remaining `{item.get('remaining')}` {item.get('unit', '')}"
            )
    else:
        lines.append("- _No evidence readiness block found._")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate Phase53 portfolio shadow evidence before manual paper review.")
    parser.add_argument("--active-registry", type=Path, default=Path("wfa_optimized_params_output/phase53_active_portfolio_shadow_registry.json"))
    parser.add_argument("--portfolio-manifest", type=Path, default=None)
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--cycle-summary", type=Path, default=Path("wfa_optimized_params_output/phase53_active_live_shadow/latest_cycle_summary.json"))
    parser.add_argument("--evidence-json", type=Path, default=Path("wfa_optimized_params_output/phase53_active_live_shadow/latest_evidence.json"))
    parser.add_argument("--out-json", type=Path, default=Path("wfa_optimized_params_output/phase53_active_live_shadow/paper_review_gate.json"))
    parser.add_argument("--out-md", type=Path, default=Path("wfa_optimized_params_output/phase53_active_live_shadow/paper_review_gate.md"))
    args = parser.parse_args()

    result = evaluate_paper_review_gate(args)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, result)
    print(
        json.dumps(
            {
                "decision": result["decision"]["status"],
                "ready_for_manual_paper_review": result["ready_for_manual_paper_review"],
                "ready_for_automated_paper": result["ready_for_automated_paper"],
                "failed_gates": result["decision"]["failed_gates"],
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
