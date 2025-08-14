from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Tuple


OUT_DIR = os.path.join(os.getcwd(), "wfa_optimized_params_output")


def find_latest(tag: str) -> str | None:
    pattern = os.path.join(OUT_DIR, f"optimized_params_*_{tag}.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    perf = data.get("is_performance", {})
    return {
        "path": path,
        "pnl_pct": perf.get("total_net_pnl_percentage"),
        "pf": perf.get("profit_factor"),
        "dd_pct": perf.get("max_drawdown_percentage"),
        "win_rate": perf.get("win_rate_percentage"),
        "trades": perf.get("num_trades"),
    }


def run_cudare_for_preset(preset: str, python_exec: str) -> None:
    print(f"[oos_compare] Running cudare.py for preset={preset} (baseline + improved)...", flush=True)
    env = os.environ.copy()
    env["RUN_BOTH"] = "1"
    env["PERIOD_PRESET"] = preset
    # Ensure output dir exists
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run([python_exec, os.path.join(os.getcwd(), "cudare.py")], env=env, check=True)


def compare_pair(preset: str) -> Tuple[Dict, Dict]:
    base_tag = f"baseline_{preset}"
    imp_tag = f"improved_{preset}"
    base_path = find_latest(base_tag)
    imp_path = find_latest(imp_tag)
    if not base_path or not imp_path:
        raise FileNotFoundError(f"Missing result JSONs for preset={preset}: baseline={base_path}, improved={imp_path}")
    base_metrics = load_metrics(base_path)
    imp_metrics = load_metrics(imp_path)
    return base_metrics, imp_metrics


def print_table(preset: str, base: Dict, imp: Dict) -> None:
    headers = ["Preset", "Profile", "PF", "PnL%", "DD%", "Win%", "Trades"]
    rows = [
        [preset, "baseline", base["pf"], base["pnl_pct"], base["dd_pct"], base["win_rate"], base["trades"]],
        [preset, "improved", imp["pf"], imp["pnl_pct"], imp["dd_pct"], imp["win_rate"], imp["trades"]],
    ]
    # Simple column widths
    colw = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r):
        return " ".join(str(v).rjust(colw[i]) for i, v in enumerate(r))
    print("\n" + fmt_row(headers))
    print(" ".join("-" * w for w in colw))
    for r in rows:
        print(fmt_row(r))
    # Verdict
    better_pf = (imp["pf"] or 0) > (base["pf"] or 0)
    better_dd = (imp["dd_pct"] or 1e9) < (base["dd_pct"] or 1e9)
    print(f"Verdict[{preset}]: PF {'↑' if better_pf else '–'}, DD {'↓' if better_dd else '–'} → {'Improved' if (better_pf or better_dd) else 'Tie/Worse'}")


def main():
    ap = argparse.ArgumentParser(description="Run OOS head-to-head for baseline vs improved using cudare.py presets.")
    ap.add_argument("--presets", nargs="*", default=["recent", "stress_july"], help="List of PERIOD_PRESET values")
    ap.add_argument("--skip-run", action="store_true", help="Only compare latest JSONs, do not run cudare")
    args = ap.parse_args()

    pyexec = sys.executable
    for preset in args.presets:
        if not args.skip_run:
            run_cudare_for_preset(preset, pyexec)
        base, imp = compare_pair(preset)
        print_table(preset, base, imp)

    print("\nDone.")


if __name__ == "__main__":
    main()

