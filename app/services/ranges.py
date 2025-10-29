from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import DataRange

_HOUR = timedelta(hours=1)


def fetch_ranges(session: Session, market: str) -> List[DataRange]:
    stmt = select(DataRange).where(DataRange.market == market).order_by(DataRange.start_timestamp)
    return list(session.scalars(stmt))


def merge_ranges(ranges: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    ordered = sorted(ranges, key=lambda r: r[0])
    if not ordered:
        return []

    merged: list[tuple[datetime, datetime]] = []
    current_start, current_end = ordered[0]

    for start, end in ordered[1:]:
        if start <= current_end + _HOUR:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def persist_ranges(session: Session, market: str, ranges: Iterable[tuple[datetime, datetime]]) -> list[DataRange]:
    session.execute(delete(DataRange).where(DataRange.market == market))
    result: list[DataRange] = []
    for start, end in ranges:
        record = DataRange(market=market, start_timestamp=start, end_timestamp=end)
        session.add(record)
        result.append(record)
    session.flush()
    return result


def add_range(session: Session, market: str, start: datetime, end: datetime) -> list[DataRange]:
    if end < start:
        raise ValueError("Range end must be >= start")

    existing_objs = fetch_ranges(session, market)
    existing = [(r.start_timestamp, r.end_timestamp) for r in existing_objs]
    merged = merge_ranges(existing + [(start, end)])

    if merged == existing:
        return existing_objs

    return persist_ranges(session, market, merged)
