from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .base import BacktestStats, MetricCalculator


class CumulativeReturnMetric(MetricCalculator):
    name = "cumulative_return"
    goal = "max"
    description = "Total return over the backtest period"

    def compute(self, stats: BacktestStats) -> float:
        if not stats.equity_curve:
            return 0.0
        return stats.final_value / stats.initial_value - 1.0


class VolatilityMetric(MetricCalculator):
    name = "volatility"
    goal = "min"
    description = "Standard deviation of periodic returns"

    def compute(self, stats: BacktestStats) -> float:
        if len(stats.returns) < 2:
            return 0.0
        return float(np.std(stats.returns, ddof=1))


class MaxDrawdownMetric(MetricCalculator):
    name = "max_drawdown"
    goal = "min"
    description = "Maximum peak-to-trough drawdown"

    def compute(self, stats: BacktestStats) -> float:
        return float(stats.max_drawdown)


class SharpeRatioMetric(MetricCalculator):
    name = "sharpe_ratio"
    goal = "max"
    description = "Sharpe ratio with zero risk-free rate"

    def compute(self, stats: BacktestStats) -> float:
        if not stats.returns:
            return 0.0
        avg = float(np.mean(stats.returns))
        std = float(np.std(stats.returns, ddof=1))
        if std == 0:
            return 0.0
        return avg / std * math.sqrt(len(stats.returns))


class WinRateMetric(MetricCalculator):
    name = "win_rate"
    goal = "max"
    description = "Proportion of winning trades"

    def compute(self, stats: BacktestStats) -> float:
        if stats.total_trades == 0:
            return 0.0
        return stats.winning_trades / stats.total_trades


METRICS: Dict[str, MetricCalculator] = {
    metric.name: metric
    for metric in [
        CumulativeReturnMetric(),
        VolatilityMetric(),
        MaxDrawdownMetric(),
        SharpeRatioMetric(),
        WinRateMetric(),
    ]
}


def get_metric(name: str) -> MetricCalculator:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown metric '{name}'") from exc


def available_metrics() -> List[MetricCalculator]:
    return list(METRICS.values())
