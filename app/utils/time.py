from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dateutil import parser


def is_isoformat(value: str) -> bool:
    try:
        parser.isoparse(value)
        return True
    except (ValueError, TypeError):
        return False


UTC = timezone.utc


def parse_iso8601(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = parser.isoparse(value)
    return ensure_utc(dt)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def to_naive_utc(dt: datetime) -> datetime:
    return ensure_utc(dt).replace(tzinfo=None)


def floor_to_hour(dt: datetime) -> datetime:
    dt = ensure_utc(dt)
    return dt.replace(minute=0, second=0, microsecond=0)


def hourly_range(start: datetime, end: datetime) -> list[datetime]:
    current = floor_to_hour(start)
    end_floor = floor_to_hour(end)
    hours: list[datetime] = []
    while current <= end_floor:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


def align_range(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start_aligned = floor_to_hour(start)
    end_aligned = floor_to_hour(end)
    if end_aligned < start_aligned:
        raise ValueError("End must be after start")
    return start_aligned, end_aligned
