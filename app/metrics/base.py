from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BacktestStats:
    returns: List[float]
    equity_curve: List[float]
    max_drawdown: float
    max_drawdown_money: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_value: float
    initial_value: float


class MetricCalculator:
    name: str
    goal: str  # "max" or "min"
    description: str

    def compute(self, stats: BacktestStats) -> float:
        raise NotImplementedError
