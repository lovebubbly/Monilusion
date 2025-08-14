from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Set


def _as_set(x: Optional[Iterable[int]]):
    return set(x) if x is not None else set()


def hour_allowed(dt: datetime, allow_hours: Optional[Iterable[int]] = None, deny_hours: Optional[Iterable[int]] = None) -> bool:
    h = int(dt.hour)
    allow = _as_set(allow_hours)
    deny = _as_set(deny_hours)
    if allow and h not in allow:
        return False
    if deny and h in deny:
        return False
    return True


def weekday_allowed(dt: datetime, allow_days: Optional[Iterable[int]] = None, deny_days: Optional[Iterable[int]] = None) -> bool:
    d = int(dt.weekday())
    allow = set(allow_days) if allow_days else set()
    deny = set(deny_days) if deny_days else set()
    if allow and d not in allow:
        return False
    if deny and d in deny:
        return False
    return True


def is_trade_allowed_by_time(
    timestamp: datetime,
    allowed_hours: Iterable[int] | Set[int] = (10, 12, 14, 17),
    blocked_days: Iterable[str] | Set[str] = ("Monday",),
) -> bool:
    """
    거래가 허용된 시간대 및 요일인지 확인.
    - allowed_hours: 허용된 시간(hour) 집합 (예: {10, 12, 14, 17})
    - blocked_days: 차단할 요일 문자열 집합 (예: {'Monday'})
    """
    if not hour_allowed(timestamp, allowed_hours, None):
        return False
    dayname = timestamp.strftime('%A')
    if str(dayname).strip() in {str(x).strip() for x in blocked_days}:
        return False
    return True
