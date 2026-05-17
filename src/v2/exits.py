# -*- coding: utf-8 -*-
# src/v2/exits.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExitResult:
    exit_index: int
    exit_price: float
    reason: str
    partial_exit_price: float | None = None
    partial_fraction: float = 0.0

    def blended_exit_price(self) -> float:
        if self.partial_exit_price is None or self.partial_fraction <= 0.0:
            return float(self.exit_price)
        f = min(1.0, max(0.0, float(self.partial_fraction)))
        return float(f * self.partial_exit_price + (1.0 - f) * self.exit_price)


class HybridExit:
    """
    Time stop + ATR trailing + RR target + break-even-after-R.

    ``intrabar_policy`` controls OHLC bars where SL and TP are both touched:
    ``conservative`` resolves stop first; ``optimistic`` resolves target first.
    """

    def __init__(
        self,
        sl_atr_mult=2.6,
        rr=3.0,
        atr_trail_mult=1.0,
        time_stop_bars=48,
        fee_bps=2.0,
        be_after_r: float = 1.0,
        use_trailing: bool = True,
        tp1_r: float | None = None,
        tp1_frac: float = 0.5,
        intrabar_policy: str = "conservative",
    ):
        self.sl_k = float(sl_atr_mult)
        self.rr = float(rr)
        self.trail_k = float(atr_trail_mult)
        self.time_stop = int(time_stop_bars)
        self.fee_bps = float(fee_bps)
        self.be_after_r = float(be_after_r)
        self.use_trailing = bool(use_trailing)
        self.tp1_r = None if tp1_r is None else float(tp1_r)
        self.tp1_frac = max(0.0, min(1.0, float(tp1_frac)))
        if intrabar_policy not in {"conservative", "optimistic"}:
            raise ValueError("intrabar_policy must be 'conservative' or 'optimistic'")
        self.intrabar_policy = intrabar_policy

    def _legacy_fee_adjusted_exit(self, direction: int, entry_price: float, raw_exit_price: float) -> float:
        fee_per_unit = entry_price * (self.fee_bps / 10000.0)
        if direction == 1:
            return raw_exit_price - fee_per_unit
        return raw_exit_price + fee_per_unit

    def _resolve_stop_target(
        self,
        direction: int,
        bar_high: float,
        bar_low: float,
        stop_price: float,
        target_price: float,
    ) -> tuple[bool, float, str]:
        if direction == 1:
            hit_stop = bar_low <= stop_price
            hit_target = bar_high >= target_price
        else:
            hit_stop = bar_high >= stop_price
            hit_target = bar_low <= target_price

        if hit_stop and hit_target:
            if self.intrabar_policy == "optimistic":
                return True, target_price, "TP_AMBIGUOUS"
            return True, stop_price, "SL_AMBIGUOUS"
        if hit_stop:
            return True, stop_price, "SL"
        if hit_target:
            return True, target_price, "TP"
        return False, 0.0, "NONE"

    def _tp1_hit(self, direction: int, bar_high: float, bar_low: float, tp1: float) -> bool:
        return bar_high >= tp1 if direction == 1 else bar_low <= tp1

    def _apply_favorable_updates(
        self,
        direction: int,
        entry_price: float,
        sl_dist: float,
        atr_value: float,
        bar_high: float,
        bar_low: float,
        current_stop: float,
        best_price: float,
        be_armed: bool,
    ) -> tuple[float, float]:
        if direction == 1:
            if be_armed and bar_high >= entry_price + self.be_after_r * sl_dist:
                current_stop = max(current_stop, entry_price)
            if self.use_trailing:
                best_price = max(best_price, bar_high)
                current_stop = max(current_stop, best_price - self.trail_k * atr_value)
        else:
            if be_armed and bar_low <= entry_price - self.be_after_r * sl_dist:
                current_stop = min(current_stop, entry_price)
            if self.use_trailing:
                best_price = min(best_price, bar_low)
                current_stop = min(current_stop, best_price + self.trail_k * atr_value)
        return current_stop, best_price

    def apply(
        self,
        i_entry: int,
        direction: int,
        entry_price: float,
        atr: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        *,
        first_bar_offset: int = 1,
        sl_dist_override: float | None = None,
        return_details: bool = False,
    ):
        """
        direction: +1 long, -1 short.

        ``first_bar_offset=0`` is for next-bar-open fills where exits may happen
        inside the entry candle. ``1`` preserves close-fill behavior.
        """
        sl_dist = float(sl_dist_override) if sl_dist_override is not None else self.sl_k * atr[i_entry]
        tp_dist = self.rr * sl_dist
        be_armed = (self.be_after_r is not None) and (self.be_after_r > 0.0)

        if direction == 1:
            stop = entry_price - sl_dist
            target = entry_price + tp_dist
            best = entry_price
            tp1 = None if self.tp1_r is None else entry_price + self.tp1_r * sl_dist
        else:
            stop = entry_price + sl_dist
            target = entry_price - tp_dist
            best = entry_price
            tp1 = None if self.tp1_r is None else entry_price - self.tp1_r * sl_dist

        tp1_price = None
        f = self.tp1_frac
        start = i_entry + max(0, int(first_bar_offset))

        for j in range(start, len(close)):
            bar_high = high[j]
            bar_low = low[j]

            if self.intrabar_policy == "optimistic":
                if tp1 is not None and tp1_price is None and self._tp1_hit(direction, bar_high, bar_low, tp1):
                    tp1_price = tp1
                    stop = max(stop, entry_price) if direction == 1 else min(stop, entry_price)
                stop, best = self._apply_favorable_updates(
                    direction, entry_price, sl_dist, atr[j], bar_high, bar_low, stop, best, be_armed
                )

            exited, exit_price, reason = self._resolve_stop_target(direction, bar_high, bar_low, stop, target)
            if exited:
                result = ExitResult(j, float(exit_price), reason, tp1_price, f if tp1_price is not None else 0.0)
                if return_details:
                    return result
                return j, self._legacy_fee_adjusted_exit(direction, entry_price, result.blended_exit_price())

            if self.intrabar_policy == "conservative":
                if tp1 is not None and tp1_price is None and self._tp1_hit(direction, bar_high, bar_low, tp1):
                    tp1_price = tp1
                    stop = max(stop, entry_price) if direction == 1 else min(stop, entry_price)
                stop, best = self._apply_favorable_updates(
                    direction, entry_price, sl_dist, atr[j], bar_high, bar_low, stop, best, be_armed
                )

            if j - i_entry >= self.time_stop:
                result = ExitResult(j, float(close[j]), "TIME_STOP", tp1_price, f if tp1_price is not None else 0.0)
                if return_details:
                    return result
                return j, self._legacy_fee_adjusted_exit(direction, entry_price, result.blended_exit_price())

        result = ExitResult(len(close) - 1, float(close[-1]), "EOD", tp1_price, f if tp1_price is not None else 0.0)
        if return_details:
            return result
        return result.exit_index, self._legacy_fee_adjusted_exit(direction, entry_price, result.blended_exit_price())
