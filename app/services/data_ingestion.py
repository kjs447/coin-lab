from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List

import requests
from dateutil import parser
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import session_scope
from ..models import OHLCV
from ..schemas import DataIngestionResponse, TimeRange
from ..utils.time import align_range, ensure_utc, to_naive_utc
from . import ranges as range_service

_HOUR = timedelta(hours=1)


@dataclass
class IngestionSummary:
    ingested: int
    interpolated: int
    ranges: List[tuple[str, datetime, datetime]]

    def as_response(self) -> DataIngestionResponse:
        return DataIngestionResponse(
            ingested_candles=self.ingested,
            interpolated_candles=self.interpolated,
            ranges=[
                {
                    "market": market,
                    "start_timestamp": start,
                    "end_timestamp": end,
                }
                for market, start, end in self.ranges
            ],
        )


class UpbitClient:
    def __init__(
        self,
        base_url: str | None = None,
        http: requests.Session | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ):
        self.base_url = base_url or settings.upbit_base_url
        self.http = http or requests.Session()
        self.max_retries = max_retries or settings.upbit_max_retries
        self.backoff_seconds = backoff_seconds or settings.upbit_retry_backoff

    def fetch_hour_candles(self, market: str, to: datetime, count: int) -> list[dict]:
        params = {
            "market": market,
            "to": ensure_utc(to).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": count,
        }
        attempt = 0
        while True:
            response = self.http.get(
                f"{self.base_url}/candles/minutes/60", params=params, timeout=10
            )
            status = response.status_code
            if status == 429 and attempt + 1 < self.max_retries:
                time.sleep(self.backoff_seconds * (attempt + 1))
                attempt += 1
                continue
            if 500 <= status < 600 and attempt + 1 < self.max_retries:
                time.sleep(self.backoff_seconds * (attempt + 1))
                attempt += 1
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                raise RuntimeError(
                    f"Upbit request failed with status {status}: {detail or response.reason}"
                ) from exc
            return response.json()


class DataIngestionService:
    def __init__(self, client: UpbitClient | None = None):
        self.client = client or UpbitClient()

    def ingest(self, payload: TimeRange) -> IngestionSummary:
        start_aligned, end_aligned = align_range(payload.start, payload.end)
        start_aligned = to_naive_utc(start_aligned)
        end_aligned = to_naive_utc(end_aligned)
        market = payload.market
        ingested = 0
        interpolated = 0

        with session_scope() as session:
            existing_ranges = range_service.fetch_ranges(session, market)
            missing_segments = self._compute_missing_segments(
                existing_ranges, start_aligned, end_aligned
            )

            for seg_start, seg_end in missing_segments:
                ingested += self._collect_segment(session, market, seg_start, seg_end)

            interpolated_delta, retry_ingested = self._fill_and_validate(
                session, market, start_aligned, end_aligned
            )
            interpolated += interpolated_delta
            ingested += retry_ingested
            updated_objs = range_service.add_range(
                session, market, start_aligned, end_aligned
            )
            updated_ranges = [
                (record.market, record.start_timestamp, record.end_timestamp)
                for record in updated_objs
            ]

        return IngestionSummary(ingested=ingested, interpolated=interpolated, ranges=updated_ranges)

    def _compute_missing_segments(
        self,
        ranges: Iterable[DataRange],
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        segments: list[tuple[datetime, datetime]] = []
        cursor = start
        for record in sorted(ranges, key=lambda r: r.start_timestamp):
            rng_start = record.start_timestamp
            rng_end = record.end_timestamp
            if rng_end < cursor:
                continue
            if rng_start > end:
                break
            if rng_start > cursor:
                segments.append((cursor, min(rng_start - _HOUR, end)))
            cursor = max(cursor, rng_end + _HOUR)
            if cursor > end:
                break
        if cursor <= end:
            segments.append((cursor, end))
        return [seg for seg in segments if seg[0] <= seg[1]]

    def _collect_segment(
        self, session: Session, market: str, start: datetime, end: datetime
    ) -> int:
        if end < start:
            return 0

        to_pointer = end + _HOUR
        collected: dict[datetime, dict] = {}
        while True:
            batch = self.client.fetch_hour_candles(
                market, to_pointer, settings.max_candle_request
            )
            if not batch:
                break
            for item in batch:
                candle_time = ensure_utc(
                    parser.isoparse(item["candle_date_time_utc"])
                ).replace(tzinfo=None)
                if candle_time < start:
                    continue
                if candle_time > end:
                    continue
                collected[candle_time] = item
            earliest = min(
                ensure_utc(parser.isoparse(c["candle_date_time_utc"])) for c in batch
            ).replace(tzinfo=None)
            if earliest <= start:
                break
            to_pointer = earliest - _HOUR
            if to_pointer <= start:
                break
        if not collected:
            return 0

        existing_stmt = select(OHLCV.timestamp).where(
            OHLCV.market == market,
            OHLCV.timestamp >= start,
            OHLCV.timestamp <= end,
        )
        existing = {row for row in session.scalars(existing_stmt)}

        inserted = 0
        for timestamp, candle in collected.items():
            if timestamp in existing:
                continue
            session.add(
                OHLCV(
                    market=market,
                    timestamp=timestamp,
                    opening_price=float(candle["opening_price"]),
                    high_price=float(candle["high_price"]),
                    low_price=float(candle["low_price"]),
                    trade_price=float(candle["trade_price"]),
                    candle_acc_trade_price=float(candle["candle_acc_trade_price"]),
                    candle_acc_trade_volume=float(candle["candle_acc_trade_volume"]),
                )
            )
            existing.add(timestamp)
            inserted += 1
        session.flush()
        return inserted

    def _fill_and_validate(
        self, session: Session, market: str, start: datetime, end: datetime
    ) -> tuple[int, int]:
        candles = self._load_candles(session, market, start, end)
        missing_hours = self._find_missing_hours(candles, start, end)
        retry_ingested = 0
        if missing_hours:
            retry_missing = list(missing_hours)
            for hour in retry_missing:
                retry_ingested += self._collect_segment(session, market, hour, hour)
            session.flush()
            candles = self._load_candles(session, market, start, end)
            missing_hours = self._find_missing_hours(candles, start, end)

        interpolated = 0
        if missing_hours:
            interpolated = self._interpolate(session, market, candles, missing_hours)
            candles = self._load_candles(session, market, start, end)

        self._validate_complete(candles, start, end)
        return interpolated, retry_ingested

    def _load_candles(
        self, session: Session, market: str, start: datetime, end: datetime
    ) -> dict[datetime, OHLCV]:
        stmt = (
            select(OHLCV)
            .where(OHLCV.market == market)
            .where(OHLCV.timestamp >= start)
            .where(OHLCV.timestamp <= end)
        )
        return {row.timestamp: row for row in session.scalars(stmt)}

    def _find_missing_hours(
        self, candles: dict[datetime, OHLCV], start: datetime, end: datetime
    ) -> List[datetime]:
        missing: List[datetime] = []
        current = start
        while current <= end:
            if current not in candles:
                missing.append(current)
            current += _HOUR
        return missing

    def _interpolate(
        self,
        session: Session,
        market: str,
        candles: dict[datetime, OHLCV],
        missing_hours: Iterable[datetime],
    ) -> int:
        interpolated = 0
        for ts in sorted(missing_hours):
            prev = candles.get(ts - _HOUR)
            if prev is None:
                raise RuntimeError(
                    f"Cannot interpolate {ts.isoformat()} without previous candle"
                )
            new_candle = OHLCV(
                market=market,
                timestamp=ts,
                opening_price=prev.trade_price,
                high_price=prev.trade_price,
                low_price=prev.trade_price,
                trade_price=prev.trade_price,
                candle_acc_trade_price=0.0,
                candle_acc_trade_volume=0.0,
            )
            session.add(new_candle)
            candles[ts] = new_candle
            interpolated += 1
        session.flush()
        return interpolated

    def _validate_complete(
        self, candles: dict[datetime, OHLCV], start: datetime, end: datetime
    ) -> None:
        current = start
        while current <= end:
            candle = candles.get(current)
            if candle is None:
                raise RuntimeError(f"Missing candle at {current.isoformat()} after interpolation")
            if candle.high_price < candle.low_price:
                raise RuntimeError(
                    f"Invalid candle at {current.isoformat()}: high < low"
                )
            if not (candle.low_price <= candle.opening_price <= candle.high_price):
                raise RuntimeError(
                    f"Invalid open price at {current.isoformat()}"
                )
            if not (candle.low_price <= candle.trade_price <= candle.high_price):
                raise RuntimeError(
                    f"Invalid close price at {current.isoformat()}"
                )
            current += _HOUR
