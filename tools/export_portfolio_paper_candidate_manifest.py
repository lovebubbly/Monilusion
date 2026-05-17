from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


MANUAL_REVIEW_PERMISSION = "MANUAL_PAPER_REVIEW_ONLY_NO_ORDERS"
READY_STATUS = "READY_FOR_MANUAL_PAPER_REVIEW"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _all_gates_pass(gates: list[dict[str, Any]]) -> bool:
    return bool(gates) and all(gate.get("pass") is True for gate in gates)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        _write_json(path, payload)


def _candidate_ready(paper_review_gate: dict[str, Any]) -> bool:
    decision = paper_review_gate.get("decision", {})
    return decision.get("status") == READY_STATUS and paper_review_gate.get("ready_for_manual_paper_review") is True


def export_paper_candidate_manifest(paper_review_gate: dict[str, Any], source_path: Path) -> dict[str, Any]:
    decision = paper_review_gate.get("decision", {})
    failed_gates = decision.get("failed_gates") or []
    if decision.get("status") != READY_STATUS:
        raise SystemExit(
            "Refusing to export portfolio paper candidate because paper-review gate is not READY_FOR_MANUAL_PAPER_REVIEW "
            f"(status={decision.get('status')!r}, failed_gates={failed_gates!r})."
        )
    if paper_review_gate.get("ready_for_manual_paper_review") is not True:
        raise SystemExit("Refusing to export portfolio paper candidate because manual paper review is not ready.")
    if failed_gates:
        raise SystemExit(f"Refusing to export portfolio paper candidate because paper-review gates failed: {failed_gates}.")
    if not _all_gates_pass(paper_review_gate.get("gates", [])):
        raise SystemExit("Refusing to export portfolio paper candidate because not every paper-review gate passed.")
    if paper_review_gate.get("ready_for_automated_paper") is True or paper_review_gate.get("ready_for_paper") is True:
        raise SystemExit("Refusing to export portfolio paper candidate because automated paper must remain disabled.")
    if paper_review_gate.get("paper_trading_automation") != "HOLD":
        raise SystemExit("Refusing to export portfolio paper candidate because paper_trading_automation is not HOLD.")

    return {
        "schema_version": 1,
        "manifest_type": "portfolio_paper_candidate",
        "date": date.today().isoformat(),
        "source_paper_review_gate": str(source_path),
        "active_registry": paper_review_gate.get("active_registry"),
        "active_manifest": paper_review_gate.get("active_manifest"),
        "active_manifest_sha256": paper_review_gate.get("active_manifest_sha256"),
        "validation_json": paper_review_gate.get("validation_json"),
        "cycle_summary": paper_review_gate.get("cycle_summary"),
        "evidence_json": paper_review_gate.get("evidence_json"),
        "decision": "READY_FOR_MANUAL_PAPER_REVIEW",
        "ready_for_manual_paper_review": True,
        "ready_for_automated_paper": False,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "execution_permission": MANUAL_REVIEW_PERMISSION,
        "promotion_constraints": {
            "manual_review_required": True,
            "automated_paper_orders_allowed": False,
            "live_shadow_evidence_required": True,
            "source_gate_must_remain_passing": True,
        },
        "paper_review_decision": decision,
        "evidence_readiness": paper_review_gate.get("evidence_readiness", {}),
        "funding_coverage": paper_review_gate.get("funding_coverage", {}),
        "component_signature": paper_review_gate.get("component_signature", {}),
        "execution_assumptions": paper_review_gate.get("execution_assumptions", {}),
        "source_cpu_cuda_diff_signature": paper_review_gate.get("source_cpu_cuda_diff_signature", {}),
    }


def maybe_export_paper_candidate_manifest(
    *,
    paper_review_gate: dict[str, Any],
    source_path: Path,
    out_path: Path,
    out_md: Path | None = None,
    status_out: Path | None = None,
    skip_if_not_ready: bool = False,
    clear_stale_on_skip: bool = False,
) -> dict[str, Any]:
    decision = paper_review_gate.get("decision", {})
    failed_gates = decision.get("failed_gates") or []
    if skip_if_not_ready and not _candidate_ready(paper_review_gate):
        cleared_paths: list[str] = []
        if clear_stale_on_skip:
            for candidate_path in [out_path, out_md]:
                if candidate_path is not None and candidate_path.exists():
                    candidate_path.unlink()
                    cleared_paths.append(str(candidate_path))
        status = {
            "schema_version": 1,
            "mode": "portfolio_paper_candidate_export",
            "status": "SKIPPED_NOT_READY",
            "source_paper_review_gate": str(source_path),
            "paper_review_status": decision.get("status"),
            "paper_review_failed_gates": failed_gates,
            "ready_for_manual_paper_review": paper_review_gate.get("ready_for_manual_paper_review") is True,
            "ready_for_automated_paper": paper_review_gate.get("ready_for_automated_paper") is True,
            "ready_for_paper": paper_review_gate.get("ready_for_paper") is True,
            "paper_trading_automation": paper_review_gate.get("paper_trading_automation"),
            "execution_permission": paper_review_gate.get("execution_permission"),
            "candidate_out": str(out_path),
            "candidate_out_md": str(out_md) if out_md is not None else None,
            "cleared_stale_paths": cleared_paths,
            "rationale": "Paper-review gate is not ready; no manual paper candidate was exported.",
        }
        _write_status(status_out, status)
        return status

    manifest = export_paper_candidate_manifest(paper_review_gate, source_path)
    _write_json(out_path, manifest)
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, manifest)
    status = {
        "schema_version": 1,
        "mode": "portfolio_paper_candidate_export",
        "status": "EXPORTED_MANUAL_REVIEW_CANDIDATE",
        "source_paper_review_gate": str(source_path),
        "paper_review_status": decision.get("status"),
        "paper_review_failed_gates": failed_gates,
        "ready_for_manual_paper_review": True,
        "ready_for_automated_paper": False,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "execution_permission": MANUAL_REVIEW_PERMISSION,
        "candidate_out": str(out_path),
        "candidate_out_md": str(out_md) if out_md is not None else None,
        "active_manifest_sha256": manifest.get("active_manifest_sha256"),
    }
    _write_status(status_out, status)
    return status


def _write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Paper Candidate Manifest",
        "",
        f"- Decision: `{manifest['decision']}`",
        f"- Ready for manual paper review: `{str(manifest['ready_for_manual_paper_review']).lower()}`",
        f"- Ready for automated paper: `false`",
        f"- Ready for paper: `false`",
        f"- Paper trading automation: `{manifest['paper_trading_automation']}`",
        f"- Execution permission: `{manifest['execution_permission']}`",
        f"- Active manifest SHA-256: `{manifest.get('active_manifest_sha256')}`",
        "",
        "This artifact is a manual paper-review candidate only. It is not an order-routing configuration.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a manual-review-only portfolio paper candidate manifest from a passing Phase53 paper-review gate."
    )
    parser.add_argument("--paper-review-gate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--status-out", type=Path, default=None)
    parser.add_argument(
        "--skip-if-not-ready",
        action="store_true",
        help="Return a successful SKIPPED_NOT_READY status instead of failing when the paper gate is still HOLD.",
    )
    parser.add_argument(
        "--clear-stale-on-skip",
        action="store_true",
        help="When used with --skip-if-not-ready, remove stale current candidate outputs if the gate is not ready.",
    )
    args = parser.parse_args()

    source_path = _resolve(args.paper_review_gate)
    out_path = _resolve(args.out)
    status = maybe_export_paper_candidate_manifest(
        paper_review_gate=_load_json(source_path),
        source_path=source_path,
        out_path=out_path,
        out_md=_resolve(args.out_md) if args.out_md else None,
        status_out=_resolve(args.status_out) if args.status_out else None,
        skip_if_not_ready=args.skip_if_not_ready,
        clear_stale_on_skip=args.clear_stale_on_skip,
    )
    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
