from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return round(float(statistics.median(finite)), 4) if finite else 0.0


def _candidate_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in obj.get("candidates", []) if "full_sample_cpu_reference" in row]


def _fold_key(fold: dict[str, Any]) -> tuple[str, str]:
    return (str(fold.get("test_start", ""))[:10], str(fold.get("test_end", ""))[:10])


def _fold_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        for fold in candidate.get("fixed_candidate_wfo", {}).get("folds", []):
            buckets[_fold_key(fold)].append(fold)

    rows = []
    for (start, end), folds in buckets.items():
        returns = [_num(fold.get("summary", {}).get("return_pct")) for fold in folds]
        pfs = [_num(fold.get("summary", {}).get("net_pf")) for fold in folds]
        mdds = [_num(fold.get("summary", {}).get("mdd_pct")) for fold in folds]
        trades = [_num(fold.get("summary", {}).get("trades")) for fold in folds]
        pass_count = sum(1 for fold in folds if fold.get("pass"))
        count = len(folds)
        rows.append(
            {
                "test_start": start,
                "test_end": end,
                "candidates": count,
                "pass_ratio": round(pass_count / count, 4) if count else 0.0,
                "median_return_pct": _median(returns),
                "median_net_pf": _median(pfs),
                "median_mdd_pct": _median(mdds),
                "median_trades": _median(trades),
            }
        )
    rows.sort(key=lambda row: (row["pass_ratio"], row["median_return_pct"], row["median_net_pf"]))
    return rows


def _top_candidates(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        summary = candidate["full_sample_cpu_reference"]["summary"]
        failed = candidate.get("decision", {}).get("failed_gates", [])
        intrabar = candidate.get("intrabar_policy_band", {})
        intrabar_delta = intrabar.get("delta_optimistic_minus_conservative", {})
        intrabar_conservative = intrabar.get("conservative", {})
        rows.append(
            {
                "rank": candidate["rank"],
                "param_id": candidate["param_id"],
                "failed_gates": failed,
                "num_failed_gates": len(failed),
                "return_pct": _num(summary.get("return_pct")),
                "net_pf": _num(summary.get("net_pf")),
                "mdd_pct": _num(summary.get("mdd_pct")),
                "trades": int(summary.get("trades", 0)),
                "wfo_pass_ratio": _num(candidate.get("fixed_candidate_wfo", {}).get("pass_ratio")),
                "dsr": _num(candidate.get("dsr", {}).get("dsr")),
                "mc_return_p05": _num(candidate.get("monte_carlo", {}).get("return_pct_p05")),
                "cost_stress_pass_ratio": _num(candidate.get("cost_stress", {}).get("pass_ratio")),
                "intrabar_ambiguous_exits": int(intrabar_conservative.get("ambiguous_intrabar_exits", 0)),
                "intrabar_return_gap_pct": _num(intrabar_delta.get("return_pct")),
                "intrabar_pf_gap": _num(intrabar_delta.get("net_pf")),
            }
        )
    rows.sort(
        key=lambda row: (
            row["num_failed_gates"],
            -row["wfo_pass_ratio"],
            -row["return_pct"],
            -row["net_pf"],
            row["mdd_pct"],
        )
    )
    return rows[:limit]


def _gate_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts.update(candidate.get("decision", {}).get("failed_gates", []))
    return dict(counts)


def _write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Validation Manifest Analysis",
        "",
        f"- Source: `{analysis['source_manifest']}`",
        f"- Decision: `{analysis['decision']}`",
        f"- Validated candidates: {analysis['validated_candidates']}",
        f"- Promoted to shadow: {analysis['promoted_to_shadow']}",
        f"- Candidate universe: raw={analysis['candidate_universe'].get('raw_candidates')}, unique={analysis['candidate_universe'].get('unique_effective_candidates')}",
        "",
        "## Failed Gates",
    ]
    for name, count in analysis["failed_gate_counts"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## Top Candidates"])
    for row in analysis["top_candidates"]:
        lines.append(
            "- "
            f"rank {row['rank']}: return={row['return_pct']}%, pf={row['net_pf']}, "
            f"mdd={row['mdd_pct']}%, wfo={row['wfo_pass_ratio']}, dsr={row['dsr']}, "
            f"stress={row['cost_stress_pass_ratio']}, "
            f"intrabar_ambiguous={row['intrabar_ambiguous_exits']}, "
            f"intrabar_gap={row['intrabar_return_gap_pct']}%, failed={row['failed_gates']}  "
            f"`{row['param_id']}`"
        )

    lines.extend(["", "## Weakest WFO Folds"])
    for row in analysis["weakest_wfo_folds"]:
        lines.append(
            "- "
            f"{row['test_start']} to {row['test_end']}: pass={row['pass_ratio']}, "
            f"median_return={row['median_return_pct']}%, median_pf={row['median_net_pf']}, "
            f"median_mdd={row['median_mdd_pct']}%, median_trades={row['median_trades']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a batch validation manifest and summarize weak folds.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--weak-folds", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    obj = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = _candidate_rows(obj)
    folds = _fold_summary(candidates)
    analysis = {
        "source_manifest": str(args.manifest),
        "decision": obj.get("summary", {}).get("decision"),
        "validated_candidates": len(candidates),
        "promoted_to_shadow": obj.get("summary", {}).get("promoted_to_shadow", 0),
        "candidate_universe": obj.get("candidate_universe", {}),
        "topk_cscv_pbo": obj.get("topk_cscv_pbo", {}),
        "failed_gate_counts": _gate_counts(candidates),
        "top_candidates": _top_candidates(candidates, args.top),
        "weakest_wfo_folds": folds[: args.weak_folds],
        "all_wfo_folds": folds,
    }

    text = json.dumps(analysis, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.out_md, analysis)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
