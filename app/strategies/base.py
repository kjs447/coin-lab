from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type

import backtrader as bt


@dataclass
class HyperParameter:
    name: str
    default: float
    kind: str  # "int" or "float"
    recommended_min: float | None = None
    recommended_max: float | None = None
    description: str | None = None


@dataclass
class StrategySpec:
    identifier: str
    name: str
    description: str
    parameters: List[HyperParameter]
    strategy_cls: Type[bt.Strategy]


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, StrategySpec] = {}

    def register(self, spec: StrategySpec) -> None:
        self._registry[spec.identifier] = spec

    def all(self) -> List[StrategySpec]:
        return list(self._registry.values())

    def get(self, identifier: str) -> StrategySpec:
        try:
            return self._registry[identifier]
        except KeyError as exc:
            raise ValueError(f"Unknown strategy '{identifier}'") from exc


registry = StrategyRegistry()
