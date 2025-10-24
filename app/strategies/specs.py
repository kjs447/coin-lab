from __future__ import annotations

import backtrader as bt

from .base import HyperParameter, StrategySpec, registry


class WindowAwareStrategy(bt.Strategy):
    params = dict(live_start=None)

    def should_skip(self) -> bool:
        live_start = self.p.live_start
        if live_start is None:
            return False
        current_dt = bt.num2date(self.data.datetime[0]).replace(tzinfo=None)
        return current_dt < live_start


class MovingAverageCrossStrategy(WindowAwareStrategy):
    params = dict(live_start=None, short_period=12, long_period=26)

    def __init__(self) -> None:
        short = bt.indicators.SimpleMovingAverage(self.data.close, period=int(self.params.short_period))
        long = bt.indicators.SimpleMovingAverage(self.data.close, period=int(self.params.long_period))
        self.crossover = bt.indicators.CrossOver(short, long)

    def next(self) -> None:
        if self.should_skip():
            return
        if not self.position and self.crossover[0] > 0:
            self.order_target_percent(target=1.0)
        elif self.position and self.crossover[0] < 0:
            self.order_target_percent(target=0.0)


class BollingerRsiReversionStrategy(WindowAwareStrategy):
    params = dict(
        live_start=None,
        bollinger_period=20,
        devfactor=2.0,
        rsi_period=14,
        rsi_low=30.0,
        rsi_high=70.0,
    )

    def __init__(self) -> None:
        self.bbands = bt.indicators.BollingerBands(
            self.data.close,
            period=int(self.params.bollinger_period),
            devfactor=float(self.params.devfactor),
        )
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=int(self.params.rsi_period)
        )

    def next(self) -> None:
        if self.should_skip():
            return
        close = self.data.close[0]
        rsi = self.rsi[0]
        if not self.position:
            if close < self.bbands.lines.bot[0] and rsi < float(self.params.rsi_low):
                self.order_target_percent(target=1.0)
        else:
            if close > self.bbands.lines.top[0] or rsi > float(self.params.rsi_high):
                self.order_target_percent(target=0.0)


class VolatilityBreakoutStrategy(WindowAwareStrategy):
    params = dict(live_start=None, multiplier=0.5)

    def next(self) -> None:
        if self.should_skip():
            return
        if len(self.data.close) < 2:
            return
        prev_high = self.data.high[-1]
        prev_low = self.data.low[-1]
        price_range = prev_high - prev_low
        upper = self.data.open[0] + price_range * float(self.params.multiplier)
        lower = self.data.open[0] - price_range * float(self.params.multiplier)
        close = self.data.close[0]

        if not self.position and close > upper:
            self.order_target_percent(target=1.0)
        elif self.position and close < lower:
            self.order_target_percent(target=0.0)


class ObvMovingAverageStrategy(WindowAwareStrategy):
    params = dict(live_start=None, short_period=10, long_period=30)

    def __init__(self) -> None:
        obv = bt.indicators.OnBalanceVolume(self.data)
        short = bt.indicators.ExponentialMovingAverage(obv, period=int(self.params.short_period))
        long = bt.indicators.ExponentialMovingAverage(obv, period=int(self.params.long_period))
        self.crossover = bt.indicators.CrossOver(short, long)

    def next(self) -> None:
        if self.should_skip():
            return
        if not self.position and self.crossover[0] > 0:
            self.order_target_percent(target=1.0)
        elif self.position and self.crossover[0] < 0:
            self.order_target_percent(target=0.0)


class WeightedCloseTrendStrategy(WindowAwareStrategy):
    params = dict(live_start=None, ema_period=15, threshold=0.01)

    def __init__(self) -> None:
        self.weighted_close = bt.indicators.WeightedClose(self.data)
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.weighted_close, period=int(self.params.ema_period)
        )

    def next(self) -> None:
        if self.should_skip():
            return
        if self.data.close[0] == 0:
            return
        ratio = (self.ema[0] - self.data.close[0]) / self.data.close[0]
        threshold = float(self.params.threshold)
        if not self.position and ratio >= threshold:
            self.order_target_percent(target=1.0)
        elif self.position and ratio <= -threshold:
            self.order_target_percent(target=0.0)


registry.register(
    StrategySpec(
        identifier="ma_cross",
        name="Moving Average Crossover",
        description="Buy on golden cross of short SMA over long SMA, sell on death cross.",
        parameters=[
            HyperParameter(
                name="short_period",
                default=12,
                kind="int",
                recommended_min=5,
                recommended_max=60,
                description="Short window for SMA",
            ),
            HyperParameter(
                name="long_period",
                default=26,
                kind="int",
                recommended_min=20,
                recommended_max=200,
                description="Long window for SMA",
            ),
        ],
        strategy_cls=MovingAverageCrossStrategy,
    )
)

registry.register(
    StrategySpec(
        identifier="bollinger_rsi",
        name="Bollinger + RSI Reversion",
        description="Contrarian entries at Bollinger extremes confirmed by RSI.",
        parameters=[
            HyperParameter("bollinger_period", 20, "int", 10, 40, "Bollinger lookback"),
            HyperParameter("devfactor", 2.0, "float", 1.0, 3.0, "Band width multiplier"),
            HyperParameter("rsi_period", 14, "int", 7, 21, "RSI lookback"),
            HyperParameter("rsi_low", 30.0, "float", 10.0, 45.0, "Oversold threshold"),
            HyperParameter("rsi_high", 70.0, "float", 55.0, 90.0, "Overbought threshold"),
        ],
        strategy_cls=BollingerRsiReversionStrategy,
    )
)

registry.register(
    StrategySpec(
        identifier="volatility_breakout",
        name="Volatility Breakout",
        description="Breakout when price exceeds open +/- prior range times multiplier.",
        parameters=[
            HyperParameter("multiplier", 0.5, "float", 0.1, 1.5, "Range multiplier k"),
        ],
        strategy_cls=VolatilityBreakoutStrategy,
    )
)

registry.register(
    StrategySpec(
        identifier="obv_ma",
        name="OBV EMA Crossover",
        description="Cross of short vs long EMA applied to OBV.",
        parameters=[
            HyperParameter("short_period", 10, "int", 3, 30, "Short EMA period"),
            HyperParameter("long_period", 30, "int", 10, 100, "Long EMA period"),
        ],
        strategy_cls=ObvMovingAverageStrategy,
    )
)

registry.register(
    StrategySpec(
        identifier="weighted_close",
        name="Weighted Close EMA",
        description="EMA of weighted close compared to price ratio triggers trend entries.",
        parameters=[
            HyperParameter("ema_period", 15, "int", 5, 60, "EMA lookback on weighted close"),
            HyperParameter("threshold", 0.01, "float", 0.002, 0.05, "Ratio threshold"),
        ],
        strategy_cls=WeightedCloseTrendStrategy,
    )
)
