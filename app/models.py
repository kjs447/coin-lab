from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class OHLCV(Base):
    __tablename__ = "ohlcv"
    id = Column(Integer, primary_key=True)
    market = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    opening_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    trade_price = Column(Float, nullable=False)
    candle_acc_trade_price = Column(Float, nullable=False)
    candle_acc_trade_volume = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint("market", "timestamp", name="uq_market_timestamp"),)


class DataRange(Base):
    __tablename__ = "data_range"
    id = Column(Integer, primary_key=True)
    market = Column(String(20), nullable=False, index=True)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint("market", "start_timestamp", "end_timestamp", name="uq_market_range"),)

    @property
    def as_tuple(self) -> tuple[datetime, datetime]:
        return self.start_timestamp, self.end_timestamp
