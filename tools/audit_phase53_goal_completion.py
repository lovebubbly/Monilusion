from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_DIR = ROOT / "wfa_optimized_params_output" / "phase53_active_live_shadow"
DEFAULT_VALIDATION = (
    ROOT
    / "wfa_optimized_params_output"
    / "phase53_current_train_refresh_20260517_active_registry_from_existing"
    / "validation_phase53_portfolio_shadow_current_train_20260517_active_registry_from_existing.json"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_map(paper_review: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not paper_review:
        return {}
    return {str(row.get("name")): row for row in paper_review.get("gates", [])}


def _gate_pass(gates: dict[str, dict[str, Any]], name: str) -> bool:
    return gates.get(name, {}).get("pass") is True


def _has_keys(mapping: dict[str, Any] | None, keys: list[str]) -> bool:
    if not mapping:
        return False
    return all(mapping.get(key) is not None for key in keys)


def _all_component_intrabar_bands(validation: dict[str, Any] | None) -> bool:
    components = (validation or {}).get("component_source_cpu_cuda_diffs", [])
    if not components:
        return False
    for component in components:
        band = component.get("intrabar_policy_comparison", {})
        if not isinstance(band.get("conservative"), dict) or not isinstance(band.get("optimistic"), dict):
            return False
    return True


def _all_component_pf_diff(validation: dict[str, Any] | None) -> bool:
    components = (validation or {}).get("component_source_cpu_cuda_diffs", [])
    if not components:
        return False
    for component in components:
        diff = component.get("diff", {})
        cpu = component.get("cpu_reference", {})
        if "net_profit_factor" not in diff or "gross_profit_factor" not in diff:
            return False
        if "net_profit_factor" not in cpu or "gross_profit_factor" not in cpu:
            return False
    return True


def _cost_stress_pass(cost_stress: dict[str, Any] | None) -> bool:
    if not cost_stress:
        return False
    if cost_stress.get("pass_ratio") == 1.0:
        return True
    scenarios = cost_stress.get("scenarios", [])
    return bool(scenarios) and all(row.get("pass") is True for row in scenarios)


def _status(passed: bool, blockers: list[str] | None = None) -> str:
    if passed:
        return "PROVEN"
    if blockers:
        return "BLOCKED"
    return "MISSING_OR_WEAK"


def _item(
    item_id: str,
    requirement: str,
    passed: bool,
    evidence: dict[str, Any],
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "requirement": requirement,
        "status": _status(passed, blockers),
        "evidence": evidence,
        "blockers": blockers or [],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve(args.manifest)
    validation_path = _resolve(args.validation_json)
    paper_path = _resolve(args.paper_review_json)
    status_path = _resolve(args.status_report_json)

    manifest = _load_json_if_exists(manifest_path)
    validation = _load_json_if_exists(validation_path)
    paper = _load_json_if_exists(paper_path)
    status = _load_json_if_exists(status_path)
    gates = _gate_map(paper)

    assumptions = (validation or {}).get("execution_assumptions", {})
    manifest_assumptions = (manifest or {}).get("assumptions", {})
    source_diff = (validation or {}).get("source_cpu_cuda_diff", {})
    portfolio_diag = (paper or {}).get("portfolio_diagnostic", {})
    diag_source = portfolio_diag.get("source", {})
    diag_active = portfolio_diag.get("active_manifest", {})
    status_evidence = (status or {}).get("stored_evidence", {})
    paper_candidate = (status or {}).get("paper_candidate_export", {})
    paper_decision = (paper or {}).get("decision", {})
    evidence_shortfalls = (status or {}).get("evidence_shortfalls", {})

    items = [
        _item(
            "market_binance_usdm",
            "Use Binance USD-M perpetual assumptions.",
            assumptions.get("market") == "Binance USD-M perpetual",
            {"market": assumptions.get("market"), "validation_json": str(validation_path)},
        ),
        _item(
            "htf_no_lookahead",
            "Verify 4H higher-timeframe alignment without lookahead.",
            "no_lookahead" in str(assumptions.get("htf_alignment", "")),
            {"htf_alignment": assumptions.get("htf_alignment"), "validation_json": str(validation_path)},
        ),
        _item(
            "next_bar_open",
            "Use next-bar-open fills by default.",
            assumptions.get("entry_price_policy") == "next_bar_open" and int(assumptions.get("entry_delay_bars", -1)) == 1,
            {
                "entry_price_policy": assumptions.get("entry_price_policy"),
                "entry_delay_bars": assumptions.get("entry_delay_bars"),
            },
        ),
        _item(
            "intrabar_policy_band",
            "Report conservative and optimistic intrabar SL/TP scenarios while gating on conservative behavior.",
            assumptions.get("intrabar_policy_base") == "conservative" and _all_component_intrabar_bands(validation),
            {
                "intrabar_policy_base": assumptions.get("intrabar_policy_base"),
                "component_comparison_count": len((validation or {}).get("component_source_cpu_cuda_diffs", [])),
            },
        ),
        _item(
            "costs_funding_slippage",
            "Reflect fee, slippage, and actual funding events for Binance futures.",
            (
                float(assumptions.get("commission_rate", 0.0)) > 0
                and float(assumptions.get("slippage_rate", 0.0)) > 0
                and assumptions.get("funding_model") == "actual_funding_events"
                and bool(assumptions.get("funding_rate_csv"))
                and ((paper or {}).get("funding_coverage") or {}).get("status") == "ok"
            ),
            {
                "commission_rate": assumptions.get("commission_rate"),
                "slippage_rate": assumptions.get("slippage_rate"),
                "funding_model": assumptions.get("funding_model"),
                "funding_rate_csv": assumptions.get("funding_rate_csv"),
                "funding_coverage": (paper or {}).get("funding_coverage"),
            },
        ),
        _item(
            "intrabar_mtm_mdd",
            "Use mark-to-market and intrabar drawdown basis.",
            assumptions.get("drawdown_basis") == "intrabar_mark_to_market_equity_curve",
            {"drawdown_basis": assumptions.get("drawdown_basis")},
        ),
        _item(
            "gross_net_pf",
            "Separate gross and net profit factor in validation and CPU/CUDA diff.",
            (
                assumptions.get("profit_factor_reporting")
                == "component_gross_and_net_pf_plus_portfolio_net_trade_pf"
                and _all_component_pf_diff(validation)
            ),
            {
                "profit_factor_reporting": assumptions.get("profit_factor_reporting"),
                "component_diff_count": len((validation or {}).get("component_source_cpu_cuda_diffs", [])),
            },
        ),
        _item(
            "cpu_cuda_diff",
            "Diff-test CPU reference engine against CUDA/fast search results.",
            (
                source_diff.get("all_pass") is True
                and source_diff.get("diff_pass_count") == source_diff.get("component_count")
                and _gate_pass(gates, "validation_source_cpu_cuda_diff_pass")
                and _gate_pass(gates, "validation_source_cpu_cuda_diff_details_match")
            ),
            {"source_cpu_cuda_diff": source_diff},
        ),
        _item(
            "wfo_cpcv_pbo",
            "Use WFO plus purged/embargoed CSCV/CPCV-style PBO diagnostics for portfolio selection.",
            (
                diag_source.get("pbo_enabled") is True
                and float(diag_source.get("pbo", 1.0)) <= 0.2
                and int(diag_source.get("pbo_samples", 0)) > 0
                and int(diag_source.get("pbo_weight_candidates", 0)) > 0
                and diag_source.get("formal_pbo_gate_pass") is True
            ),
            {
                "source_portfolio_diagnostic": diag_source.get("source_portfolio_diagnostic"),
                "pbo": diag_source.get("pbo"),
                "pbo_samples": diag_source.get("pbo_samples"),
                "pbo_weight_candidates": diag_source.get("pbo_weight_candidates"),
                "formal_pbo_gate_pass": diag_source.get("formal_pbo_gate_pass"),
            },
        ),
        _item(
            "dsr_mc_cost_stress",
            "Require DSR, Monte Carlo, and cost stress diagnostics before promotion.",
            (
                _has_keys((validation or {}).get("dsr"), ["dsr", "annualized_sharpe"])
                and _has_keys((validation or {}).get("monte_carlo"), ["return_pct_p05", "prob_return_positive"])
                and _cost_stress_pass((validation or {}).get("cost_stress"))
                and _gate_pass(gates, "manifest_portfolio_diagnostic_strict")
                and _gate_pass(gates, "source_portfolio_diagnostic_strict")
                and _gate_pass(gates, "portfolio_diagnostic_source_matches_manifest")
            ),
            {
                "validation_dsr": (validation or {}).get("dsr"),
                "validation_monte_carlo": (validation or {}).get("monte_carlo"),
                "cost_stress_pass_ratio": (validation or {}).get("cost_stress", {}).get("pass_ratio"),
                "active_portfolio_diagnostic": diag_active,
            },
        ),
        _item(
            "strict_shadow_promotion",
            "Promote only strict-passing candidates to no-order shadow logging.",
            (
                (manifest or {}).get("decision") == "PROMOTE_TO_SHADOW"
                and (manifest or {}).get("execution_permission") == "NO_ORDERS_SHADOW_LOGGING_ONLY"
                and (manifest or {}).get("ready_for_shadow") is True
                and (manifest or {}).get("ready_for_paper") is False
                and manifest_assumptions.get("paper_trading") == "disabled"
            ),
            {
                "manifest_decision": (manifest or {}).get("decision"),
                "execution_permission": (manifest or {}).get("execution_permission"),
                "ready_for_shadow": (manifest or {}).get("ready_for_shadow"),
                "ready_for_paper": (manifest or {}).get("ready_for_paper"),
                "paper_trading": manifest_assumptions.get("paper_trading"),
            },
        ),
        _item(
            "live_shadow_logging",
            "Accumulate live-only shadow evidence with manifest hash and observation-mode integrity.",
            (
                int(status_evidence.get("total_events", 0)) > 0
                and status_evidence.get("observation_modes") == ["latest_live_closed_candle"]
                and _gate_pass(gates, "evidence_live_only")
                and _gate_pass(gates, "evidence_hash_integrity")
            ),
            {
                "stored_evidence": status_evidence,
                "run_history": (status or {}).get("run_history"),
            },
        ),
        _item(
            "paper_auto_hold_until_ready",
            "Hold automated paper trading until strict paper-review evidence is ready.",
            (
                (paper or {}).get("ready_for_automated_paper") is False
                and (paper or {}).get("ready_for_paper") is False
                and (paper or {}).get("paper_trading_automation") == "HOLD"
                and _gate_pass(gates, "paper_automation_disabled")
            ),
            {
                "paper_decision": paper_decision,
                "paper_candidate_export": paper_candidate,
                "ready_for_automated_paper": (paper or {}).get("ready_for_automated_paper"),
                "ready_for_paper": (paper or {}).get("ready_for_paper"),
                "paper_trading_automation": (paper or {}).get("paper_trading_automation"),
            },
        ),
        _item(
            "manual_paper_candidate_ready",
            "Export a manual-review-only paper candidate only after shadow evidence is ready.",
            paper_decision.get("status") == "READY_FOR_MANUAL_PAPER_REVIEW",
            {
                "paper_decision": paper_decision,
                "paper_candidate_export": paper_candidate,
                "evidence_shortfalls": evidence_shortfalls,
            },
            blockers=paper_decision.get("failed_gates") or ["shadow_evidence_ready"],
        ),
    ]

    proven = sum(1 for row in items if row["status"] == "PROVEN")
    blocked = [row for row in items if row["status"] == "BLOCKED"]
    missing = [row for row in items if row["status"] == "MISSING_OR_WEAK"]
    overall_status = "COMPLETE" if proven == len(items) else "INCOMPLETE"
    if blocked and all(row["id"] == "manual_paper_candidate_ready" for row in blocked) and not missing:
        overall_status = "BLOCKED_LIVE_SHADOW_EVIDENCE"

    return {
        "schema_version": 1,
        "mode": "phase53_goal_completion_audit",
        "overall_status": overall_status,
        "complete": overall_status == "COMPLETE",
        "summary": {
            "proven": proven,
            "blocked": len(blocked),
            "missing_or_weak": len(missing),
            "total": len(items),
        },
        "source_files": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "validation_json": {"path": str(validation_path), "sha256": _sha256(validation_path)},
            "paper_review_json": {"path": str(paper_path), "sha256": _sha256(paper_path)},
            "status_report_json": {"path": str(status_path), "sha256": _sha256(status_path)},
        },
        "items": items,
        "remaining_blockers": [row for row in items if row["status"] != "PROVEN"],
        "next_action": (status or {}).get("next_action", "continue_live_no_order_shadow_logging_and_recheck_evidence"),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Phase53 Goal Completion Audit",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Complete: `{str(report['complete']).lower()}`",
        f"- Proven: `{summary['proven']}/{summary['total']}`",
        f"- Blocked: `{summary['blocked']}`",
        f"- Missing or weak: `{summary['missing_or_weak']}`",
        f"- Next action: `{report.get('next_action')}`",
        "",
        "## Matrix",
        "",
        "| Requirement | Status | Evidence | Blockers |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["items"]:
        evidence_bits = []
        evidence = row.get("evidence", {})
        for key in sorted(evidence):
            value = evidence[key]
            if isinstance(value, (dict, list)):
                continue
            evidence_bits.append(f"{key}={value}")
        evidence_text = "; ".join(evidence_bits[:4])
        blockers = ", ".join(row.get("blockers", []))
        lines.append(f"| `{row['id']}` | `{row['status']}` | {evidence_text} | {blockers} |")
    lines.extend(["", "## Remaining Blockers", ""])
    for row in report["remaining_blockers"]:
        lines.append(f"- `{row['id']}`: {row['requirement']} Blockers: `{row.get('blockers')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit current Phase53 artifacts against the active thread goal.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "wfa_optimized_params_output" / "phase53_active_portfolio_shadow_manifest.json")
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--paper-review-json", type=Path, default=ACTIVE_DIR / "paper_review_gate.json")
    parser.add_argument("--status-report-json", type=Path, default=ACTIVE_DIR / "status_report.json")
    parser.add_argument("--out-json", type=Path, default=ACTIVE_DIR / "goal_completion_matrix.json")
    parser.add_argument("--out-md", type=Path, default=ACTIVE_DIR / "goal_completion_matrix.md")
    args = parser.parse_args()

    report = audit(args)
    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, report)
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "complete": report["complete"],
                "summary": report["summary"],
                "remaining_blockers": [row["id"] for row in report["remaining_blockers"]],
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
