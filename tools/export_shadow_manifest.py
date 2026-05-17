from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


NO_ORDER_PERMISSION = "NO_ORDERS_SHADOW_LOGGING_ONLY"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = candidate.get("full_sample_cpu_reference", {}).get("summary")
    if summary:
        return summary
    return candidate.get("saved_cuda_summary", {})


def _promoted_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    promoted = []
    for candidate in manifest.get("candidates", []):
        decision = candidate.get("decision", {})
        failed = decision.get("failed_gates") or []
        if (
            candidate.get("strict_ready_for_shadow") is True
            and decision.get("status") == "PROMOTE_TO_SHADOW"
            and decision.get("ready_for_shadow") is True
            and not failed
            and candidate.get("source_period_diff_pass") is True
        ):
            promoted.append(candidate)
    return promoted


def export_shadow_manifest(validation_manifest: dict[str, Any], source_path: Path) -> dict[str, Any]:
    summary = validation_manifest.get("summary", {})
    promoted = _promoted_candidates(validation_manifest)
    if summary.get("decision") != "PROMOTE_TO_SHADOW" or int(summary.get("promoted_to_shadow", 0)) <= 0:
        raise SystemExit(
            "Refusing to export shadow manifest because validation summary is not PROMOTE_TO_SHADOW "
            f"(decision={summary.get('decision')!r}, promoted={summary.get('promoted_to_shadow')!r})."
        )
    if not promoted:
        raise SystemExit("Refusing to export shadow manifest because no candidate passed all strict promotion checks.")

    source_cuda = validation_manifest.get("source_cuda_json")
    if not source_cuda:
        raise SystemExit("Refusing to export shadow manifest because source_cuda_json is missing.")

    candidates = []
    for candidate in promoted:
        candidates.append(
            {
                "rank": int(candidate["rank"]),
                "param_id": candidate["param_id"],
                "summary": _candidate_summary(candidate),
                "wfo_pass_ratio": candidate.get("fixed_candidate_wfo", {}).get("pass_ratio"),
                "pbo": validation_manifest.get("topk_cscv_pbo", {}).get("pbo"),
                "dsr": candidate.get("dsr", {}).get("dsr"),
                "monte_carlo": candidate.get("monte_carlo"),
                "cost_stress_pass_ratio": candidate.get("cost_stress", {}).get("pass_ratio"),
                "source_period_diff_pass": candidate.get("source_period_diff_pass"),
                "intrabar_policy_band": candidate.get("intrabar_policy_band"),
                "decision": candidate["decision"],
            }
        )

    return {
        "schema_version": 2,
        "date": date.today().isoformat(),
        "source_validation_manifest": str(source_path),
        "source_cuda_results": source_cuda,
        "decision": "PROMOTE_TO_SHADOW",
        "ready_for_shadow": True,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "execution_permission": NO_ORDER_PERMISSION,
        "assumptions": validation_manifest.get("assumptions", {}),
        "criteria": validation_manifest.get("criteria", {}),
        "candidate_universe": validation_manifest.get("candidate_universe", {}),
        "topk_cscv_pbo": validation_manifest.get("topk_cscv_pbo", {}),
        "summary": {
            "validated": summary.get("validated"),
            "promoted_to_shadow": len(candidates),
            "held": summary.get("held"),
            "failure_counts": summary.get("failure_counts", {}),
            "decision": "PROMOTE_TO_SHADOW",
            "ready_for_paper": False,
        },
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a no-order shadow manifest from a strict CUDA validation manifest.")
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    source_path = _resolve(args.validation_manifest)
    manifest = export_shadow_manifest(_load_json(source_path), source_path)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "ready_for_shadow": manifest["ready_for_shadow"],
                "ready_for_paper": manifest["ready_for_paper"],
                "candidate_count": len(manifest["candidates"]),
                "execution_permission": manifest["execution_permission"],
                "out": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
