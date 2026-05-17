from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[int, dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["performance"]["combo_index"]): row for row in obj["results"]}


def _num(value: Any) -> float:
    if value == "inf":
        return float("inf")
    return float(value)


def _passes(perf: dict[str, Any], min_return: float, min_pf: float, max_mdd: float, min_trades: int) -> bool:
    pf_value = perf.get("net_profit_factor", perf.get("profit_factor"))
    return (
        _num(perf["total_net_pnl_percentage"]) >= min_return
        and _num(pf_value) >= min_pf
        and _num(perf["max_drawdown_percentage"]) <= max_mdd
        and int(perf["num_trades"]) >= min_trades
        and not bool(perf.get("error"))
    )


def _brief(row: dict[str, Any] | None) -> str:
    if row is None:
        return "not_saved"
    perf = row["performance"]
    return (
        f"pnl={perf['total_net_pnl_percentage']} "
        f"net_pf={perf.get('net_profit_factor', perf.get('profit_factor'))} "
        f"gross_pf={perf.get('gross_profit_factor', 'n/a')} "
        f"mdd={perf['max_drawdown_percentage']} "
        f"trades={perf['num_trades']} "
        f"strict={perf.get('strict_pass', 'n/a')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Join CUDA strategy result JSON files.")
    parser.add_argument("--is-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--oos-file", required=True, type=Path)
    parser.add_argument("--min-return", type=float, default=30.0)
    parser.add_argument("--min-pf", type=float, default=1.3)
    parser.add_argument("--max-mdd", type=float, default=25.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sort-by", choices=["val", "oos"], default="val")
    parser.add_argument("--require-is-pass", action="store_true")
    args = parser.parse_args()

    is_rows = _load(args.is_file)
    val_rows = _load(args.val_file)
    oos_rows = _load(args.oos_file)
    print(f"loaded: IS={len(is_rows)} VAL={len(val_rows)} OOS={len(oos_rows)}")

    candidates = []
    for combo_index, val_row in val_rows.items():
        oos_row = oos_rows.get(combo_index)
        if oos_row is None:
            continue
        is_row = is_rows.get(combo_index)
        if args.require_is_pass and (
            is_row is None
            or not _passes(is_row["performance"], args.min_return, args.min_pf, args.max_mdd, args.min_trades)
        ):
            continue
        if not _passes(val_row["performance"], args.min_return, args.min_pf, args.max_mdd, args.min_trades):
            continue
        if not _passes(oos_row["performance"], args.min_return, args.min_pf, args.max_mdd, args.min_trades):
            continue
        candidates.append((combo_index, is_row, val_row, oos_row))

    sort_label = args.sort_by.upper()
    sort_index = 2 if args.sort_by == "val" else 3
    candidates.sort(
        key=lambda item: _num(item[sort_index]["performance"]["total_net_pnl_percentage"]),
        reverse=True,
    )

    print(
        "pass VAL+OOS: "
        f"{len(candidates)} "
        f"(return>={args.min_return}, pf>={args.min_pf}, mdd<={args.max_mdd}, trades>={args.min_trades})"
    )
    for rank, (combo_index, is_row, val_row, oos_row) in enumerate(candidates[: args.top], start=1):
        print(f"\n#{rank} combo_index={combo_index} sorted_by={sort_label}")
        print(f"  IS : {_brief(is_row)}")
        print(f"  VAL: {_brief(val_row)}")
        print(f"  OOS: {_brief(oos_row)}")
        print(f"  ID : {oos_row['performance']['param_id']}")
        print(f"  params: {json.dumps(oos_row['parameters'], ensure_ascii=False, sort_keys=True)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
