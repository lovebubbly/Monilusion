from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json_default(obj: Any) -> Any:
    return str(obj)


def _score(row: dict[str, Any], rank_metric: str) -> float:
    perf = row["performance"]
    if rank_metric in {"robust", "strict_return"} and "rank_score" in perf:
        return float(perf["rank_score"])
    return float(perf.get("total_net_pnl_percentage", -math.inf))


def _latest_result(output_dir: Path, profile: str, started: float, known: set[Path]) -> Path | None:
    pattern = f"top_results_BTCUSDT_{profile}_*.json"
    candidates = []
    for path in output_dir.glob(pattern):
        if path in known:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= started - 1.0:
            candidates.append((mtime, path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _merge_results(paths: list[Path], top_k: int, rank_metric: str) -> dict[str, Any]:
    if not paths:
        raise SystemExit("No chunk results to merge.")
    objects = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = objects[0]
    rows = []
    for path, obj in zip(paths, objects):
        for row in obj.get("results", []):
            merged_row = dict(row)
            merged_row["source_chunk_file"] = str(path)
            rows.append(merged_row)
    rows.sort(key=lambda row: _score(row, rank_metric), reverse=True)
    selected = rows[:top_k]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    payload = dict(first)
    payload.update(
        {
            "search_profile": f"{first.get('search_profile')}_chunked",
            "chunked": True,
            "rank_metric": rank_metric,
            "source_chunk_files": [str(path) for path in paths],
            "num_chunk_files": len(paths),
            "results": selected,
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a CUDA search in Windows-safe parameter chunks and merge the top results."
    )
    parser.add_argument("--profile", default="phase4_robust")
    parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-08-14")
    parser.add_argument("--total-combos", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-top-k", type=int, default=50)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0)
    parser.add_argument("--strict-min-return", type=float, default=30.0)
    parser.add_argument("--strict-min-pf", type=float, default=1.3)
    parser.add_argument("--strict-max-mdd", type=float, default=25.0)
    parser.add_argument("--strict-min-trades", type=int, default=30)
    parser.add_argument("--rank-metric", default="robust", choices=["return", "strict_return", "robust"])
    parser.add_argument("--timeout-minutes-per-chunk", type=float, default=5.0)
    parser.add_argument("--output-dir", default=Path("wfa_optimized_params_output"), type=Path)
    parser.add_argument("--out", default=Path("wfa_optimized_params_output/top_results_BTCUSDT_phase4_robust_chunked.json"), type=Path)
    parser.add_argument("--keep-going", action="store_true", help="Keep later chunks running after a chunk failure.")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    known = set(output_dir.glob(f"top_results_BTCUSDT_{args.profile}_*.json"))
    chunk_paths: list[Path] = []
    failures = []
    runner = ROOT / "tools" / "run_cuda_strategy_search.py"
    num_chunks = math.ceil(args.total_combos / args.chunk_size)

    for chunk_idx, start_index in enumerate(range(0, args.total_combos, args.chunk_size), start=1):
        limit = min(args.chunk_size, args.total_combos - start_index)
        print(f"[chunk {chunk_idx}/{num_chunks}] start={start_index} limit={limit}")
        cmd = [
            sys.executable,
            str(runner),
            "--profile",
            args.profile,
            "--csv",
            args.csv,
            "--start",
            args.start,
            "--end",
            args.end,
            "--top-k",
            str(args.chunk_top_k),
            "--batch-size",
            str(limit),
            "--commission",
            str(args.commission),
            "--slippage",
            str(args.slippage),
            "--entry-delay-bars",
            str(args.entry_delay_bars),
            "--funding-rate-per-8h",
            str(args.funding_rate_per_8h),
            "--strict-min-return",
            str(args.strict_min_return),
            "--strict-min-pf",
            str(args.strict_min_pf),
            "--strict-max-mdd",
            str(args.strict_max_mdd),
            "--strict-min-trades",
            str(args.strict_min_trades),
            "--rank-metric",
            args.rank_metric,
            "--param-start-index",
            str(start_index),
            "--param-limit",
            str(limit),
            "--timeout-minutes",
            str(args.timeout_minutes_per_chunk),
        ]
        started = time.time()
        env = os.environ.copy()
        env["WFA_OUTPUT_DIR"] = str(output_dir)
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
        result_path = _latest_result(output_dir, args.profile, started, known)
        if proc.returncode != 0:
            failure = {"chunk": chunk_idx, "start": start_index, "limit": limit, "returncode": proc.returncode}
            failures.append(failure)
            print(f"[chunk {chunk_idx}/{num_chunks}] failed: {failure}")
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            tail = "\n".join(combined.strip().splitlines()[-20:])
            if tail:
                print(tail)
            if not args.keep_going:
                break
        if result_path is not None:
            known.add(result_path)
            chunk_paths.append(result_path)
            print(f"[chunk {chunk_idx}/{num_chunks}] result={result_path}")
        elif proc.returncode == 0:
            failure = {"chunk": chunk_idx, "start": start_index, "limit": limit, "returncode": 0, "reason": "no result file"}
            failures.append(failure)
            print(f"[chunk {chunk_idx}/{num_chunks}] no result file")
            if not args.keep_going:
                break

    payload = _merge_results(chunk_paths, args.top_k, args.rank_metric)
    payload["chunk_failures"] = failures
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"chunks_ok={len(chunk_paths)} failures={len(failures)} merged_top_k={len(payload['results'])}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
