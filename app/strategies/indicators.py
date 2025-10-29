from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import backtrader as bt


@dataclass
class IndicatorParameter:
    name: str
    default: float
    kind: str  # "int" or "float"
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None


class IndicatorDefinition:
    def __init__(self, name: str, description: str, parameters: List[IndicatorParameter], builder: Callable[..., bt.Indicator]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._builder = builder

    def create(self, data: bt.feeds.DataBase, **kwargs) -> bt.Indicator:
        params = {}
        for parameter in self.parameters:
            if parameter.kind == "int":
                params[parameter.name] = int(kwargs.get(parameter.name, parameter.default))
            else:
                params[parameter.name] = float(kwargs.get(parameter.name, parameter.default))
        return self._builder(data, **params)


def _sma_builder(data: bt.feeds.DataBase, period: int = 20) -> bt.Indicator:
    return bt.indicators.SimpleMovingAverage(data.close, period=period)


def _rsi_builder(data: bt.feeds.DataBase, period: int = 14) -> bt.Indicator:
    return bt.indicators.RelativeStrengthIndex(data.close, period=period)


def _bollinger_builder(
    data: bt.feeds.DataBase, period: int = 20, devfactor: float = 2.0
) -> bt.Indicator:
    return bt.indicators.BollingerBands(data.close, period=period, devfactor=devfactor)


def _volatility_builder(data: bt.feeds.DataBase, period: int = 20) -> bt.Indicator:
    return bt.indicators.StandardDeviation(data.close, period=period)


class OnBalanceVolume(bt.Indicator):
    lines = ('obv',)

    def __init__(self) -> None:
        self.addminperiod(2)

    def next(self) -> None:
        if len(self.data) <= 1:
            self.lines.obv[0] = 0.0
            return

        prev = self.lines.obv[-1]
        close = self.data.close[0]
        prev_close = self.data.close[-1]
        volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 0.0

        if close > prev_close:
            self.lines.obv[0] = prev + volume
        elif close < prev_close:
            self.lines.obv[0] = prev - volume
        else:
            self.lines.obv[0] = prev


def _obv_builder(data: bt.feeds.DataBase) -> bt.Indicator:
    return OnBalanceVolume(data)


class WeightedClose(bt.Indicator):
    lines = ('wc',)

    def next(self) -> None:
        high = float(self.data.high[0]) if hasattr(self.data, 'high') else float(self.data[0])
        low = float(self.data.low[0]) if hasattr(self.data, 'low') else float(self.data[0])
        close = float(self.data.close[0]) if hasattr(self.data, 'close') else float(self.data[0])
        self.lines.wc[0] = (high + low + 2 * close) / 4.0


def _weighted_close_builder(data: bt.feeds.DataBase) -> bt.Indicator:
    return bt.indicators.WeightedClose(data)


INDICATORS: Dict[str, IndicatorDefinition] = {
    "sma": IndicatorDefinition(
        name="Simple Moving Average",
        description="Simple moving average of close prices",
        parameters=[IndicatorParameter("period", 20, "int", "Number of candles")],
        builder=_sma_builder,
    ),
    "rsi": IndicatorDefinition(
        name="Relative Strength Index",
        description="Momentum oscillator measuring overbought/oversold levels",
        parameters=[IndicatorParameter("period", 14, "int", "Number of candles")],
        builder=_rsi_builder,
    ),
    "bollinger": IndicatorDefinition(
        name="Bollinger Bands",
        description="Bollinger Bands using SMA and standard deviation",
        parameters=[
            IndicatorParameter("period", 20, "int", "Basis period"),
            IndicatorParameter("devfactor", 2.0, "float", "Standard deviation multiplier"),
        ],
        builder=_bollinger_builder,
    ),
    "volatility": IndicatorDefinition(
        name="Volatility (StdDev)",
        description="Standard deviation of closing prices",
        parameters=[IndicatorParameter("period", 20, "int", "Lookback period")],
        builder=_volatility_builder,
    ),
    "obv": IndicatorDefinition(
        name="On-Balance Volume",
        description="Cumulative volume indicator",
        parameters=[],
        builder=_obv_builder,
    ),
    "weighted_close": IndicatorDefinition(
        name="Weighted Close",
        description="(High + Low + 2 * Close) / 4",
        parameters=[],
        builder=_weighted_close_builder,
    ),
}


def get_indicator(name: str) -> IndicatorDefinition:
    try:
        return INDICATORS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown indicator '{name}'") from exc
