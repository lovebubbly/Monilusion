from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any


NO_ORDER_PERMISSION = "NO_ORDERS_SHADOW_LOGGING_ONLY"
ALLOWED_REMAINING_FAILURES = {"formal_portfolio_export"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path, base: Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else base / out


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _assert_portfolio_ready_for_shadow_source(portfolio: dict[str, Any]) -> None:
    gate = portfolio.get("formal_promotion_gate", {})
    failed = set(gate.get("failed_gates") or [])
    unexpected = sorted(failed - ALLOWED_REMAINING_FAILURES)
    if unexpected:
        raise SystemExit(f"Refusing export because portfolio still has non-export failed gates: {unexpected}.")
    if gate.get("pbo_gate_pass") is not True:
        raise SystemExit("Refusing export because portfolio PBO gate did not pass.")
    if gate.get("meta_gate_pass") is not True:
        raise SystemExit("Refusing export because portfolio meta-selection gate did not pass.")
    if gate.get("stress_gate_pass") is not True:
        raise SystemExit("Refusing export because portfolio cost-stress gate did not pass.")
    if _num(portfolio.get("portfolio_weight_pbo", {}).get("pbo"), 1.0) > 0.5:
        raise SystemExit("Refusing export because portfolio PBO is above the strict threshold.")
    meta_dsr = portfolio.get("portfolio_meta_selection", {}).get("summary", {}).get("dsr", {}).get("dsr")
    if _num(meta_dsr, 0.0) < 0.95:
        raise SystemExit("Refusing export because portfolio meta-selection DSR is below 0.95.")
    if _num(portfolio.get("portfolio", {}).get("dsr", {}).get("dsr"), 0.0) < 0.95:
        raise SystemExit("Refusing export because fixed portfolio DSR is below 0.95.")


def _latest_selected_fold(wfo: dict[str, Any]) -> dict[str, Any]:
    folds = [row for row in wfo.get("folds", []) if row.get("selected_parameters") or row.get("selected_ensemble")]
    if not folds:
        raise SystemExit(f"No selected fold found in WFO manifest {wfo.get('profile')!r}.")
    return folds[-1]


def _selection_items(fold: dict[str, Any]) -> list[dict[str, Any]]:
    if fold.get("selected_ensemble"):
        return list(fold["selected_ensemble"])
    if fold.get("selected_parameters"):
        return [
            {
                "rank": fold.get("selected_rank"),
                "param_id": fold.get("selected_param_id"),
                "parameters": fold.get("selected_parameters"),
                "performance": fold.get("selected_train_performance", {}),
            }
        ]
    return []


def _cuda_selection_items(cuda_obj: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = list(cuda_obj.get("results", []))[:limit]
    if len(rows) < limit:
        raise SystemExit(f"CUDA result has only {len(rows)} rows but {limit} are required.")
    out = []
    for row in rows:
        performance = row.get("performance", {})
        out.append(
            {
                "rank": row.get("rank"),
                "param_id": performance.get("param_id") or row.get("param_id"),
                "parameters": row.get("parameters"),
                "performance": performance,
            }
        )
    return out


def _component_rows(
    portfolio: dict[str, Any],
    source_path: Path,
    *,
    selection_basis: str,
    current_cuda_results: list[Path] | None = None,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    sources = list(portfolio.get("sources", []))
    if selection_basis == "current_cuda_result":
        if not current_cuda_results or len(current_cuda_results) != len(sources):
            raise SystemExit("--current-cuda-result must be provided once for each portfolio source.")
    for source_index, source in enumerate(portfolio.get("sources", []), start=1):
        source_file = source.get("file")
        if not source_file:
            raise SystemExit("Portfolio source is missing file.")
        wfo_path = _resolve(source_file, source_path.parent)
        wfo = _load_json(wfo_path)
        fold = _latest_selected_fold(wfo)
        selected_fold_items = _selection_items(fold)
        if not selected_fold_items:
            raise SystemExit(f"WFO source {wfo_path} has no selected parameters in its latest selected fold.")
        source_cuda_path = None
        source_cuda_obj: dict[str, Any] | None = None
        if selection_basis == "latest_wfo_fold":
            selections = selected_fold_items
        elif selection_basis == "current_cuda_result":
            source_cuda_path = _resolve(current_cuda_results[source_index - 1], Path.cwd())
            source_cuda_obj = _load_json(source_cuda_path)
            expected_profile = wfo.get("profile", source.get("profile"))
            actual_profile = source_cuda_obj.get("search_profile")
            if expected_profile and actual_profile and expected_profile != actual_profile:
                raise SystemExit(
                    f"CUDA profile mismatch for source {source_index}: expected {expected_profile!r}, got {actual_profile!r}."
                )
            selections = _cuda_selection_items(source_cuda_obj, len(selected_fold_items))
        else:
            raise SystemExit(f"Unknown selection basis: {selection_basis}")

        source_weight = _num(source.get("weight"))
        if source_weight <= 0.0:
            raise SystemExit(f"Source weight must be positive for {source_file}.")
        per_component_weight = source_weight / len(selections)
        for component_index, selection in enumerate(selections, start=1):
            params = selection.get("parameters")
            if not isinstance(params, dict):
                raise SystemExit(f"Selected component in {wfo_path} is missing parameters.")
            component_id = f"s{source_index:02d}c{component_index:02d}"
            components.append(
                {
                    "component_id": component_id,
                    "source_index": source_index,
                    "component_index": component_index,
                    "source_weight": round(source_weight, 8),
                    "component_weight": round(per_component_weight, 8),
                    "source_wfo_manifest": str(wfo_path),
                    "source_cuda_results": str(source_cuda_path) if source_cuda_path is not None else None,
                    "source_profile": wfo.get("profile", source.get("profile")),
                    "selector_mode": wfo.get("assumptions", {}).get("selector_mode", source.get("selector_mode")),
                    "fold_mode": wfo.get("assumptions", {}).get("fold_mode", source.get("fold_mode")),
                    "selection_basis": selection_basis,
                    "selection_fold": {
                        "fold": fold.get("fold"),
                        "train_start": fold.get("train_start"),
                        "train_end": fold.get("train_end"),
                        "test_start": fold.get("test_start"),
                        "test_end": fold.get("test_end"),
                    },
                    "current_cuda_selection": {
                        "period_start": source_cuda_obj.get("period_start") if source_cuda_obj else None,
                        "period_end": source_cuda_obj.get("period_end") if source_cuda_obj else None,
                        "rank_metric": source_cuda_obj.get("rank_metric") if source_cuda_obj else None,
                        "total_param_combinations": source_cuda_obj.get("total_param_combinations") if source_cuda_obj else None,
                    }
                    if source_cuda_obj
                    else None,
                    "rank": selection.get("rank"),
                    "param_id": selection.get("param_id") or selection.get("performance", {}).get("param_id"),
                    "parameters": params,
                    "selected_train_performance": selection.get("performance", {}),
                    "decision": {
                        "status": "PORTFOLIO_COMPONENT_SHADOW",
                        "ready_for_shadow": True,
                        "ready_for_paper": False,
                        "failed_gates": [],
                        "rationale": "Component is exported only as part of the Phase53 portfolio no-order shadow manifest.",
                    },
                }
            )
    total = sum(_num(row.get("component_weight")) for row in components)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Component weights must sum to 1.0, got {total}.")
    return components


def export_portfolio_shadow_manifest(
    portfolio: dict[str, Any],
    source_path: Path,
    *,
    selection_basis: str,
    current_cuda_results: list[Path] | None = None,
) -> dict[str, Any]:
    _assert_portfolio_ready_for_shadow_source(portfolio)
    components = _component_rows(
        portfolio,
        source_path,
        selection_basis=selection_basis,
        current_cuda_results=current_cuda_results,
    )
    source_assumptions = {}
    if components:
        source_assumptions = _load_json(Path(components[0]["source_wfo_manifest"])).get("assumptions", {})
    return {
        "schema_version": 3,
        "manifest_type": "portfolio_shadow",
        "date": date.today().isoformat(),
        "source_portfolio_diagnostic": str(source_path),
        "decision": "PROMOTE_TO_SHADOW",
        "ready_for_shadow": True,
        "ready_for_paper": False,
        "paper_trading_automation": "HOLD",
        "execution_permission": NO_ORDER_PERMISSION,
        "shadow_mode": "NO_ORDER_COMPONENT_SIGNAL_LOGGING",
        "portfolio": {
            "label": portfolio.get("label"),
            "allocation": "fixed_weight_split",
            "component_count": len(components),
            "source_count": len(portfolio.get("sources", [])),
            "weights": [row["component_weight"] for row in components],
            "fixed_weight_summary": portfolio.get("portfolio", {}).get("summary", {}),
            "fixed_weight_dsr": portfolio.get("portfolio", {}).get("dsr", {}),
            "monte_carlo": portfolio.get("portfolio", {}).get("monte_carlo", {}),
            "cost_stress_pass_ratio": portfolio.get("cost_stress", {}).get("pass_ratio"),
            "portfolio_weight_pbo": portfolio.get("portfolio_weight_pbo", {}),
            "portfolio_meta_selection": portfolio.get("portfolio_meta_selection", {}),
        },
        "assumptions": {
            "market": "Binance USD-M perpetual",
            "commission_rate": source_assumptions.get("commission_rate", 0.0005),
            "slippage_rate": source_assumptions.get("slippage_rate", 0.0002),
            "funding_rate_per_8h": source_assumptions.get("funding_rate_per_8h", 0.0),
            "funding_model": source_assumptions.get("funding_model", "actual_funding_events"),
            "funding_rate_csv": source_assumptions.get("funding_rate_csv"),
            "entry_delay_bars": source_assumptions.get("entry_delay_bars", 1),
            "initial_balance": 10_000.0,
            "intrabar_policy_base": "conservative",
            "shadow_execution": "no_orders_log_signals_only",
            "component_parameter_basis": selection_basis,
            "paper_trading": "disabled",
        },
        "summary": {
            "decision": "PROMOTE_TO_SHADOW",
            "ready_for_paper": False,
            "paper_trading_automation": "HOLD",
            "component_count": len(components),
            "source_count": len(portfolio.get("sources", [])),
            "remaining_failed_gates_after_export": [],
        },
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a no-order shadow manifest from a validated Phase53 portfolio diagnostic.")
    parser.add_argument("--portfolio-diagnostic", required=True, type=Path)
    parser.add_argument(
        "--selection-basis",
        choices=["latest_wfo_fold", "current_cuda_result"],
        default="latest_wfo_fold",
        help="Choose component params from latest WFO fold or from current full-period CUDA result files.",
    )
    parser.add_argument(
        "--current-cuda-result",
        action="append",
        type=Path,
        default=[],
        help="CUDA result JSON, supplied once per portfolio source when --selection-basis=current_cuda_result.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    source_path = _resolve(args.portfolio_diagnostic, Path.cwd())
    manifest = export_portfolio_shadow_manifest(
        _load_json(source_path),
        source_path,
        selection_basis=args.selection_basis,
        current_cuda_results=args.current_cuda_result,
    )
    out_path = _resolve(args.out, Path.cwd())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "ready_for_shadow": manifest["ready_for_shadow"],
                "ready_for_paper": manifest["ready_for_paper"],
                "component_count": len(manifest["components"]),
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
