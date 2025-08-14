from __future__ import annotations

"""
Lightweight parameter sweep harness.

This module outlines a CLI to iterate over combinations of Phase-1 knobs
and call into a user-provided `evaluate_strategy(params)` function.

Usage:
    python -m src.backtest.param_sweep \
        --allow-hours 10 12 14 17 \
        --deny-hours 3 4 5 6 7 8 9 15 16 \
        --deny-days 0 \
        --atr-period 14 21 \
        --min-move-pct 1.5 2.0 2.5 \
        --adx-min 20 25 \
        --ema-pairs 10:50 20:100 \
        --atr-mult 1.5 2.0 \
        --time-stop 360 720 1440 \
        --max-hold 2880 \
        --loss-streak 2 3 4

Integrate by implementing `evaluate_strategy(params: dict) -> dict` in your
codebase and importing here; or override via `--module` to import a call target.
"""

import argparse
import importlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Tuple


def _pair_int(text: str) -> Tuple[int, int]:
    a, b = text.split(":", 1)
    return int(a), int(b)


def _product(grid: Dict[str, Iterable[Any]]):
    keys = list(grid.keys())
    for values in itertools.product(*[list(grid[k]) for k in keys]):
        yield dict(zip(keys, values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="module:function path for evaluate_strategy")
    ap.add_argument("--allow-hours", nargs="*", type=int, default=[10, 12, 14, 17])
    ap.add_argument("--deny-hours", nargs="*", type=int, default=[3, 4, 5, 6, 7, 8, 9, 15, 16])
    ap.add_argument("--allow-days", nargs="*", type=int, default=[])
    ap.add_argument("--deny-days", nargs="*", type=int, default=[0])  # Monday

    ap.add_argument("--atr-period", nargs="*", type=int, default=[14, 21])
    ap.add_argument("--min-move-pct", nargs="*", type=float, default=[1.5, 2.0, 2.5])
    ap.add_argument("--adx-min", nargs="*", type=float, default=[20.0, 25.0])
    ap.add_argument("--ema-pairs", nargs="*", type=_pair_int, default=[(10, 50), (20, 100)])

    ap.add_argument("--atr-mult", nargs="*", type=float, default=[1.5, 2.0])
    ap.add_argument("--time-stop", nargs="*", type=int, default=[360, 720, 1440])
    ap.add_argument("--max-hold", nargs="*", type=int, default=[2880])

    ap.add_argument("--loss-streak", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--cooldown", nargs="*", type=float, default=[0.5])
    ap.add_argument("--day-cap", nargs="*", type=float, default=[2.0])
    ap.add_argument("--week-cap", nargs="*", type=float, default=[5.0])

    args = ap.parse_args()

    # Resolve evaluate function
    eval_fn = None
    if args.module:
        mod_path, fn_name = args.module.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        eval_fn = getattr(mod, fn_name)

    grid = {
        "allow_hours": [tuple(args.allow_hours)],
        "deny_hours": [tuple(args.deny_hours)],
        "allow_days": [tuple(args.allow_days)],
        "deny_days": [tuple(args.deny_days)],
        "atr_period": args.atr_period,
        "min_move_pct": args.min_move_pct,
        "adx_min": args.adx_min,
        "ema_pair": args.ema_pairs,
        "atr_mult": args.atr_mult,
        "time_stop": args.time_stop,
        "max_hold": args.max_hold,
        "loss_streak": args.loss_streak,
        "cooldown": args.cooldown,
        "day_cap": args.day_cap,
        "week_cap": args.week_cap,
    }

    best = None
    best_metric = float("-inf")
    for params in _product(grid):
        # Normalize params into a single dict for the strategy
        fparams = {
            "time": {
                "allow_hours": params["allow_hours"],
                "deny_hours": params["deny_hours"],
                "allow_days": params["allow_days"],
                "deny_days": params["deny_days"],
            },
            "volatility": {
                "atr_period": params["atr_period"],
                "min_move_pct": params["min_move_pct"],
            },
            "regime": {
                "adx_min": params["adx_min"],
                "ema_fast": params["ema_pair"][0],
                "ema_slow": params["ema_pair"][1],
            },
            "exits": {
                "atr_mult": params["atr_mult"],
                "time_stop_min": params["time_stop"],
                "max_hold_min": params["max_hold"],
            },
            "sizing": {
                "loss_streak_k": params["loss_streak"],
                "cooldown_factor": params["cooldown"],
                "day_loss_cap_r": params["day_cap"],
                "week_loss_cap_r": params["week_cap"],
            },
        }

        if eval_fn is None:
            # Stub mode: just print params; user plugs their evaluate function later.
            result = {"profit_factor": None, "win_rate": None, "max_dd": None}
        else:
            result = eval_fn(fparams)

        metric = _score_result(result)
        if metric > best_metric:
            best_metric = metric
            best = (fparams, result)

        print(json.dumps({"params": fparams, "result": result}, ensure_ascii=False))

    if best is not None:
        print("BEST:")
        print(json.dumps({"params": best[0], "result": best[1]}, ensure_ascii=False))


def _score_result(res: dict) -> float:
    """Heuristic scorer: prioritize profit factor, then win rate, penalize drawdown."""
    if not res or res.get("profit_factor") is None:
        return float("-inf")
    pf = float(res.get("profit_factor", 0.0))
    wr = float(res.get("win_rate", 0.0)) / 100.0
    dd = float(res.get("max_dd", 1.0))
    return pf * 1.0 + wr * 0.5 - (dd / 100.0) * 0.5


if __name__ == "__main__":
    main()

