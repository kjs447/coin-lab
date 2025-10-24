from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    market: str = Field(..., description="Market symbol, e.g. KRW-BTC")
    start: datetime = Field(..., description="Inclusive UTC start timestamp")
    end: datetime = Field(..., description="Inclusive UTC end timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "market": "KRW-BTC",
                "start": "2020-09-01T00:00:00Z",
                "end": "2020-09-10T00:00:00Z",
            }
        }


class DataRangeSchema(BaseModel):
    market: str
    start_timestamp: datetime
    end_timestamp: datetime


class DataIngestionResponse(BaseModel):
    ingested_candles: int
    interpolated_candles: int
    ranges: List[DataRangeSchema]


class StrategyParameter(BaseModel):
    name: str
    default: float
    kind: str
    recommended_min: Optional[float] = None
    recommended_max: Optional[float] = None
    description: Optional[str] = None


class StrategyInfo(BaseModel):
    identifier: str
    name: str
    description: str
    parameters: List[StrategyParameter]


class HyperparameterRange(BaseModel):
    name: str
    min: Optional[float] = None
    max: Optional[float] = None
    default: Optional[float] = None


class ExperimentInterval(BaseModel):
    start: datetime
    end: datetime


class ExperimentRequest(BaseModel):
    market: str
    strategy_id: str
    validation: ExperimentInterval
    test: ExperimentInterval
    window_hours: int = Field(..., gt=0)
    validation_windows: int = Field(..., gt=0)
    test_windows: int = Field(..., gt=0)
    parameter_ranges: List[HyperparameterRange] = []


class SensitivityPoint(BaseModel):
    value: float
    metric_value: float


class ParameterSensitivity(BaseModel):
    parameter: str
    points: List[SensitivityPoint]


class OptimizationOutcomeSchema(BaseModel):
    metric: str
    goal: str
    parameters: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    sensitivity: List[ParameterSensitivity]


class ExperimentResponse(BaseModel):
    strategy_id: str
    outcomes: List[OptimizationOutcomeSchema]
