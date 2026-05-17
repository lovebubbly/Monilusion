from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_portfolio_shadow_evidence import evaluate_shadow_evidence  # noqa: E402
from evaluate_portfolio_paper_review_gate import evaluate_paper_review_gate  # noqa: E402
from export_portfolio_paper_candidate_manifest import (  # noqa: E402
    MANUAL_REVIEW_PERMISSION,
    export_paper_candidate_manifest,
    maybe_export_paper_candidate_manifest,
)
from run_live_portfolio_shadow_poll import NO_ORDER_PERMISSION, manifest_fingerprint  # noqa: E402


def _write_manifest(path: Path) -> None:
    diagnostic = path.with_name("portfolio_diagnostic.json")
    manifest = {
        "manifest_type": "portfolio_shadow",
        "schema_version": 3,
        "source_portfolio_diagnostic": str(diagnostic),
        "decision": "PROMOTE_TO_SHADOW",
        "execution_permission": NO_ORDER_PERMISSION,
        "ready_for_shadow": True,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "portfolio": {
            "component_count": 1,
            "source_count": 1,
            "fixed_weight_summary": {
                "return_pct": 40.0,
                "mdd_pct": 10.0,
                "trades": 50,
                "trade_profit_factor": 1.5,
            },
            "fixed_weight_dsr": {"dsr": 0.96},
            "monte_carlo": {"return_pct_p05": 5.0, "prob_return_positive": 1.0},
            "cost_stress_pass_ratio": 1.0,
            "portfolio_weight_pbo": {
                "enabled": True,
                "pbo": 0.2,
                "samples": 20,
                "weight_candidates": 200,
                "score_metric": "dsr",
                "median_test_percentile": 0.8,
            },
            "portfolio_meta_selection": {
                "enabled": True,
                "summary": {
                    "return_pct": 35.0,
                    "mdd_pct": 9.0,
                    "dsr": {"dsr": 0.96},
                    "monte_carlo": {"return_pct_p05": 3.0},
                },
                "gates": [
                    {"name": "meta_return_pct", "pass": True},
                    {"name": "meta_mdd_pct", "pass": True},
                    {"name": "meta_dsr", "pass": True},
                    {"name": "meta_mc_return_p05", "pass": True},
                ],
            },
        },
        "assumptions": {
            "market": "Binance USD-M perpetual",
            "commission_rate": 0.0005,
            "slippage_rate": 0.0002,
            "funding_rate_per_8h": 0.0001,
            "funding_model": "actual_funding_events",
            "funding_rate_csv": str(path.with_name("funding.csv")),
            "entry_delay_bars": 1,
            "intrabar_policy_base": "conservative",
            "shadow_execution": "no_orders_log_signals_only",
            "paper_trading": "disabled",
        },
        "components": [
            {
                "component_id": "c1",
                "component_weight": 1.0,
                "source_profile": "test_profile",
                "source_cuda_results": str(path.with_name("cuda.json")),
                "rank": 1,
                "param_id": "test-param",
                "parameters": {},
                "decision": {
                    "ready_for_shadow": True,
                    "ready_for_paper": False,
                    "failed_gates": [],
                },
            }
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    diagnostic.write_text(
        json.dumps(
            {
                "sources": [{"file": "source.json"}],
                "formal_promotion_gate": {
                    "status": "HOLD_AUTOMATED_PAPER",
                    "ready_for_shadow": False,
                    "ready_for_paper": False,
                    "failed_gates": ["formal_portfolio_export"],
                    "pbo_gate_pass": True,
                    "meta_gate_pass": True,
                    "stress_gate_pass": True,
                },
                "portfolio": {
                    "summary": {
                        "return_pct": 40.0,
                        "mdd_pct": 10.0,
                        "trades": 50,
                        "trade_profit_factor": 1.5,
                    },
                    "dsr": {"dsr": 0.96},
                    "monte_carlo": {"return_pct_p05": 5.0, "prob_return_positive": 1.0},
                },
                "cost_stress": {"pass_ratio": 1.0},
                "portfolio_weight_pbo": {
                    "enabled": True,
                    "pbo": 0.2,
                    "samples": 20,
                    "weight_candidates": 200,
                    "score_metric": "dsr",
                    "median_test_percentile": 0.8,
                },
                "portfolio_meta_selection": {
                    "enabled": True,
                    "summary": {
                        "return_pct": 35.0,
                        "mdd_pct": 9.0,
                        "dsr": {"dsr": 0.96},
                        "monte_carlo": {"return_pct_p05": 3.0},
                    },
                    "gates": [
                        {"name": "meta_return_pct", "pass": True},
                        {"name": "meta_mdd_pct", "pass": True},
                        {"name": "meta_dsr", "pass": True},
                        {"name": "meta_mc_return_p05", "pass": True},
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _args(manifest: Path, events: Path) -> argparse.Namespace:
    return argparse.Namespace(
        portfolio_manifest=manifest,
        events_jsonl=events,
        min_observed_days=1.0,
        min_unique_signal_times=2,
        min_total_events=2,
        min_accepted_events=1,
        max_duplicate_events=0,
        max_missing_components=0,
        max_missing_manifest_hash_events=0,
        max_mismatched_manifest_hash_events=0,
        max_missing_observation_mode_events=0,
        max_non_live_evidence_events=0,
    )


def test_evidence_readiness_reports_remaining_live_shadow_requirements(tmp_path):
    manifest = tmp_path / "manifest.json"
    events = tmp_path / "events.jsonl"
    _write_manifest(manifest)
    manifest_hash = manifest_fingerprint(manifest)
    events.write_text(
        json.dumps(
            {
                "event_id": "e1",
                "component_id": "c1",
                "signal_time": "2026-01-01T00:00:00Z",
                "status": "no_entry_event",
                "manifest_sha256": manifest_hash,
                "observation_mode": "latest_live_closed_candle",
                "live_evidence_eligible": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_shadow_evidence(_args(manifest, events))

    assert result["decision"]["status"] == "SHADOW_EVIDENCE_INSUFFICIENT"
    assert result["readiness"]["next_action"] == "continue_live_no_order_shadow_logging_and_recheck_evidence"
    assert result["readiness"]["shortfalls"]["observed_days"]["remaining"] == 1.0
    assert result["readiness"]["shortfalls"]["unique_signal_times"]["remaining"] == 1
    assert result["readiness"]["shortfalls"]["total_events"]["remaining"] == 1
    assert result["readiness"]["shortfalls"]["accepted_events"]["remaining"] == 1
    assert result["readiness"]["shortfalls"]["mismatched_manifest_hash_events"]["excess"] == 0
    assert result["readiness"]["projection"]["events_per_full_live_poll"] == 1
    assert result["readiness"]["projection"]["additional_full_live_polls_for_total_events"] == 1
    assert result["readiness"]["projection"]["additional_unique_signal_times_required"] == 1
    assert result["readiness"]["projection"]["additional_accepted_events_required"] == 1
    assert result["readiness"]["projection"]["earliest_signal_time_for_observed_days"] == "2026-01-02T00:00:00+00:00"


def test_evidence_readiness_blocks_wrong_manifest_hash(tmp_path):
    manifest = tmp_path / "manifest.json"
    events = tmp_path / "events.jsonl"
    _write_manifest(manifest)
    events.write_text(
        json.dumps(
            {
                "event_id": "e1",
                "component_id": "c1",
                "signal_time": "2026-01-01T00:00:00Z",
                "status": "accepted",
                "manifest_sha256": "not-the-active-manifest",
                "observation_mode": "latest_live_closed_candle",
                "live_evidence_eligible": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_shadow_evidence(_args(manifest, events))

    assert "manifest_hash_match" in result["decision"]["failed_gates"]
    mismatch = result["readiness"]["shortfalls"]["mismatched_manifest_hash_events"]
    assert mismatch["observed"] == 1
    assert mismatch["excess"] == 1


def test_evidence_readiness_blocks_historical_replay_events(tmp_path):
    manifest = tmp_path / "manifest.json"
    events = tmp_path / "events.jsonl"
    _write_manifest(manifest)
    manifest_hash = manifest_fingerprint(manifest)
    rows = [
        {
            "event_id": "live",
            "component_id": "c1",
            "signal_time": "2026-01-01T00:00:00Z",
            "status": "no_entry_event",
            "manifest_sha256": manifest_hash,
            "observation_mode": "latest_live_closed_candle",
            "live_evidence_eligible": True,
        },
        {
            "event_id": "replay",
            "component_id": "c1",
            "signal_time": "2026-01-02T00:00:00Z",
            "status": "accepted",
            "manifest_sha256": manifest_hash,
            "observation_mode": "historical_holdout_replay_not_live_evidence",
            "live_evidence_eligible": False,
        },
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = evaluate_shadow_evidence(_args(manifest, events))

    assert "live_evidence_only" in result["decision"]["failed_gates"]
    assert result["summary"]["non_live_evidence_count"] == 1
    assert result["readiness"]["shortfalls"]["non_live_evidence_events"]["excess"] == 1


def _write_portfolio_paper_gate_fixture(
    tmp_path: Path,
    *,
    evidence_ready: bool,
    cycle_hash: str | None = None,
    train_end: str = "2025-12-31T00:00:00Z",
    signal_time: str = "2026-01-01T00:00:00Z",
    validation_param_id: str = "test-param",
    non_live_evidence_count: int = 0,
):
    manifest = tmp_path / "manifest.json"
    registry = tmp_path / "registry.json"
    validation = tmp_path / "validation.json"
    evidence = tmp_path / "evidence.json"
    cycle = tmp_path / "cycle.json"
    _write_manifest(manifest)
    manifest_hash = manifest_fingerprint(manifest)
    registry.write_text(
        json.dumps(
            {
                "mode": "phase53_active_portfolio_shadow_manifest",
                "execution_permission": NO_ORDER_PERMISSION,
                "ready_for_paper": False,
                "paper_trading_automation": "HOLD",
                "active_manifest": str(manifest),
                "active_manifest_sha256": manifest_hash,
                "source_manifest": str(manifest),
                "source_manifest_sha256": manifest_hash,
                "validation_json": str(validation),
                "train_end": train_end,
                "decision": "CURRENT_TRAIN_REFRESH_READY_FOR_SHADOW",
            }
        ),
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps(
            {
                "source_manifest": str(manifest),
                "execution_assumptions": {
                    "market": "Binance USD-M perpetual",
                    "commission_rate": 0.0005,
                    "slippage_rate": 0.0002,
                    "funding_rate_per_8h": 0.0001,
                    "funding_model": "actual_funding_events",
                    "funding_rate_csv": str(manifest.with_name("funding.csv")),
                    "entry_delay_bars": 1,
                    "entry_price_policy": "next_bar_open",
                    "intrabar_policy_base": "conservative",
                    "drawdown_basis": "intrabar_mark_to_market_equity_curve",
                    "profit_factor_reporting": "component_gross_and_net_pf_plus_portfolio_net_trade_pf",
                    "cpu_reference_engine": "tools.diff_cuda_cpu_reference.cpu_reference_backtest",
                    "htf_alignment": "4h_label_right_closed_left_ffill_with_runtime_no_lookahead_check",
                },
                "source_cpu_cuda_diff": {
                    "component_count": 1,
                    "diff_pass_count": 1,
                    "all_pass": True,
                    "failed_components": [],
                    "tolerances": {"pnl_pct": 0.01, "pf": 0.01, "mdd_pct": 0.01, "trades": 0},
                },
                "component_source_cpu_cuda_diffs": [
                    {
                        "component_id": "c1",
                        "source_profile": "test_profile",
                        "source_cuda_results": str(manifest.with_name("cuda.json")),
                        "rank": 1,
                        "param_id": "test-param",
                        "diff_pass": True,
                        "diff": {
                            "total_net_pnl_percentage": {"abs_diff": 0.0},
                            "num_trades": {"abs_diff": 0.0},
                            "net_profit_factor": {"abs_diff": 0.0},
                            "gross_profit_factor": {"abs_diff": 0.0},
                            "max_drawdown_percentage": {"abs_diff": 0.0},
                        },
                        "intrabar_policy_comparison": {
                            "conservative": {"return_pct": 1.0},
                            "optimistic": {"return_pct": 1.0},
                        },
                    }
                ],
                "decision": {
                    "status": "CURRENT_COMPONENTS_VALIDATED_FOR_SHADOW_OBSERVATION",
                    "ready_for_shadow": True,
                    "ready_for_paper": False,
                    "failed_gates": [],
                },
                "components": [
                    {
                        "component_id": "c1",
                        "weight": 1.0,
                        "source_profile": "test_profile",
                        "rank": 1,
                        "param_id": validation_param_id,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_hash,
                "ready_for_paper": False,
                "paper_trading_automation": "HOLD",
                "summary": {
                    "missing_manifest_hash_count": 0,
                    "mismatched_manifest_hash_count": 0,
                    "missing_observation_mode_count": 0,
                    "non_live_evidence_count": non_live_evidence_count,
                    "event_observation_modes": (
                        ["latest_live_closed_candle"]
                        if non_live_evidence_count == 0
                        else ["historical_holdout_replay_not_live_evidence", "latest_live_closed_candle"]
                    ),
                    "first_signal_time": signal_time,
                    "last_signal_time": signal_time,
                },
                "readiness": {
                    "shortfalls": {
                        "observed_days": {"observed": 0.0, "threshold": 14.0, "remaining": 14.0, "unit": "days"},
                        "unique_signal_times": {"observed": 1, "threshold": 24, "remaining": 23, "unit": "signal_times"},
                        "total_events": {"observed": 6, "threshold": 84, "remaining": 78, "unit": "events"},
                        "accepted_events": {"observed": 0, "threshold": 1, "remaining": 1, "unit": "accepted_events"},
                    }
                },
                "decision": {
                    "status": "SHADOW_EVIDENCE_READY_FOR_MANUAL_PAPER_REVIEW"
                    if evidence_ready
                    else "SHADOW_EVIDENCE_INSUFFICIENT",
                    "ready_for_manual_paper_review": evidence_ready,
                    "failed_gates": [] if evidence_ready else ["observed_days"],
                },
            }
        ),
        encoding="utf-8",
    )
    cycle.write_text(
        json.dumps(
            {
                "observation_mode": "latest_live_closed_candle",
                "ready_for_paper": False,
                "paper_trading_automation": "HOLD",
                "funding_coverage": {"status": "ok"},
                "polls": [
                    {
                        "manifest_sha256": cycle_hash or manifest_hash,
                        "target_signal_time": signal_time,
                        "funding_coverage": {"status": "ok"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        active_registry=registry,
        portfolio_manifest=None,
        validation_json=None,
        evidence_json=evidence,
        cycle_summary=cycle,
    )
    return args


def test_portfolio_paper_review_gate_holds_when_shadow_evidence_is_short(tmp_path):
    result = evaluate_paper_review_gate(_write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=False))

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "shadow_evidence_ready" in result["decision"]["failed_gates"]
    assert result["ready_for_automated_paper"] is False
    assert result["paper_trading_automation"] == "HOLD"


def test_portfolio_paper_review_gate_requires_cycle_manifest_hash_match(tmp_path):
    result = evaluate_paper_review_gate(
        _write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True, cycle_hash="wrong-hash")
    )

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "cycle_manifest_hash_matches" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_evidence_after_train_end(tmp_path):
    result = evaluate_paper_review_gate(
        _write_portfolio_paper_gate_fixture(
            tmp_path,
            evidence_ready=True,
            train_end="2026-01-02T00:00:00Z",
            signal_time="2026-01-01T00:00:00Z",
        )
    )

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "cycle_target_after_train_end" in result["decision"]["failed_gates"]
    assert "evidence_after_train_end" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_validation_components_to_match_manifest(tmp_path):
    result = evaluate_paper_review_gate(
        _write_portfolio_paper_gate_fixture(
            tmp_path,
            evidence_ready=True,
            validation_param_id="different-param",
        )
    )

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "validation_components_match_manifest" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_live_only_evidence(tmp_path):
    result = evaluate_paper_review_gate(
        _write_portfolio_paper_gate_fixture(
            tmp_path,
            evidence_ready=True,
            non_live_evidence_count=1,
        )
    )

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "evidence_live_only" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_validation_execution_assumptions(tmp_path):
    args = _write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True)
    validation = Path(args.validation_json or tmp_path / "validation.json")
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["execution_assumptions"]["entry_delay_bars"] = 0
    validation.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_paper_review_gate(args)

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "validation_execution_assumptions_match_manifest" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_source_cpu_cuda_diff(tmp_path):
    args = _write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True)
    validation = Path(args.validation_json or tmp_path / "validation.json")
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["source_cpu_cuda_diff"]["all_pass"] = False
    payload["source_cpu_cuda_diff"]["failed_components"] = ["c1"]
    validation.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_paper_review_gate(args)

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "validation_source_cpu_cuda_diff_pass" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_source_cpu_cuda_diff_details(tmp_path):
    args = _write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True)
    validation = Path(args.validation_json or tmp_path / "validation.json")
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["component_source_cpu_cuda_diffs"][0]["param_id"] = "different-param"
    validation.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_paper_review_gate(args)

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "validation_source_cpu_cuda_diff_details_match" in result["decision"]["failed_gates"]


def test_portfolio_paper_review_gate_requires_portfolio_diagnostic(tmp_path):
    args = _write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    diagnostic = Path(manifest["source_portfolio_diagnostic"])
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    payload["portfolio_weight_pbo"]["pbo"] = 0.75
    diagnostic.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_paper_review_gate(args)

    assert result["decision"]["status"] == "HOLD_PAPER_REVIEW"
    assert "source_portfolio_diagnostic_strict" in result["decision"]["failed_gates"]
    assert "portfolio_diagnostic_source_matches_manifest" in result["decision"]["failed_gates"]


def test_portfolio_paper_candidate_export_refuses_hold_gate(tmp_path):
    gate = evaluate_paper_review_gate(_write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=False))

    try:
        export_paper_candidate_manifest(gate, tmp_path / "paper_review_gate.json")
    except SystemExit as exc:
        assert "not READY_FOR_MANUAL_PAPER_REVIEW" in str(exc)
    else:
        raise AssertionError("paper candidate export should refuse HOLD_PAPER_REVIEW gates")


def test_portfolio_paper_candidate_maybe_export_skips_and_clears_hold_gate(tmp_path):
    gate = evaluate_paper_review_gate(_write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=False))
    out = tmp_path / "paper_candidate.json"
    out_md = tmp_path / "paper_candidate.md"
    status_out = tmp_path / "paper_candidate_status.json"
    out.write_text("stale", encoding="utf-8")
    out_md.write_text("stale", encoding="utf-8")

    status = maybe_export_paper_candidate_manifest(
        paper_review_gate=gate,
        source_path=tmp_path / "paper_review_gate.json",
        out_path=out,
        out_md=out_md,
        status_out=status_out,
        skip_if_not_ready=True,
        clear_stale_on_skip=True,
    )

    assert status["status"] == "SKIPPED_NOT_READY"
    assert status["paper_review_status"] == "HOLD_PAPER_REVIEW"
    assert status["paper_review_failed_gates"] == ["shadow_evidence_ready"]
    assert not out.exists()
    assert not out_md.exists()
    written_status = json.loads(status_out.read_text(encoding="utf-8"))
    assert written_status["status"] == "SKIPPED_NOT_READY"


def test_portfolio_paper_candidate_export_is_manual_review_only(tmp_path):
    gate = evaluate_paper_review_gate(_write_portfolio_paper_gate_fixture(tmp_path, evidence_ready=True))

    manifest = export_paper_candidate_manifest(gate, tmp_path / "paper_review_gate.json")

    assert manifest["manifest_type"] == "portfolio_paper_candidate"
    assert manifest["decision"] == "READY_FOR_MANUAL_PAPER_REVIEW"
    assert manifest["ready_for_manual_paper_review"] is True
    assert manifest["ready_for_automated_paper"] is False
    assert manifest["ready_for_paper"] is False
    assert manifest["paper_trading_automation"] == "HOLD"
    assert manifest["execution_permission"] == MANUAL_REVIEW_PERMISSION
