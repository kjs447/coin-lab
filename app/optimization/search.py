from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from ..config import settings
from ..strategies.base import HyperParameter


@dataclass
class SearchResult:
    parameters: Dict[str, float]
    score: float


ConstraintFn = Callable[[Dict[str, float]], float]


class ParameterSampler:
    def __init__(self, parameters: Iterable[HyperParameter], seed: int | None = None) -> None:
        self.parameters = list(parameters)
        self.random = random.Random(seed or settings.random_seed)

    def sample(self) -> Dict[str, float]:
        sampled: Dict[str, float] = {}
        for param in self.parameters:
            lower = param.recommended_min if param.recommended_min is not None else param.default * 0.5
            upper = param.recommended_max if param.recommended_max is not None else param.default * 1.5
            if lower == upper:
                value = lower
            elif param.kind == "int":
                value = self.random.randint(int(math.floor(lower)), int(math.ceil(upper)))
            else:
                value = self.random.uniform(lower, upper)
            sampled[param.name] = value
        return sampled


def random_search(
    parameters: Iterable[HyperParameter],
    trials: int,
    objective: Callable[[Dict[str, float]], float],
    goal: str,
    seed: int | None = None,
    constraints: Sequence[ConstraintFn] | None = None,
) -> SearchResult:
    sampler = ParameterSampler(parameters, seed)
    best_params: Dict[str, float] | None = None
    best_score: float | None = None
    accepted_trials = 0

    def satisfies(candidate: Dict[str, float]) -> bool:
        if not constraints:
            return True
        return all(func(candidate) >= 0 for func in constraints)

    attempts = 0
    max_attempts = max(trials * 50, 500)
    while accepted_trials < trials and attempts < max_attempts:
        attempts += 1
        candidate = sampler.sample()
        if not satisfies(candidate):
            continue
        score = objective(candidate)
        if best_score is None:
            best_params, best_score = candidate, score
        else:
            if goal == "max" and score > best_score:
                best_params, best_score = candidate, score
            elif goal == "min" and score < best_score:
                best_params, best_score = candidate, score
        accepted_trials += 1

    if best_params is None or best_score is None:
        raise RuntimeError("Random search produced no candidates")

    return SearchResult(parameters=best_params, score=best_score)


def slsqp_optimize(
    initial: Dict[str, float],
    parameters: Iterable[HyperParameter],
    objective: Callable[[Dict[str, float]], float],
    goal: str,
    constraints: Sequence[ConstraintFn] | None = None,
) -> SearchResult:
    param_list = list(parameters)
    keys = [param.name for param in param_list]
    bounds: List[Tuple[float, float]] = []
    integer_keys = {param.name for param in param_list if param.kind == "int"}

    for param in param_list:
        lower = param.recommended_min if param.recommended_min is not None else param.default * 0.5
        upper = param.recommended_max if param.recommended_max is not None else param.default * 1.5
        bounds.append((lower, upper))

    x0 = np.array([initial[key] for key in keys], dtype=float)

    def to_candidate(vector: np.ndarray) -> Dict[str, float]:
        candidate: Dict[str, float] = {}
        for idx, key in enumerate(keys):
            value = float(vector[idx])
            lower, upper = bounds[idx]
            value = max(lower, min(upper, value))
            if key in integer_keys:
                value = round(value)
            candidate[key] = value
        return candidate

    def wrapped(x: np.ndarray) -> float:
        candidate = to_candidate(x)
        score = objective(candidate)
        return -score if goal == "max" else score

    scipy_constraints = []
    if constraints:
        for func in constraints:
            def _constraint(x: np.ndarray, _func: ConstraintFn = func) -> float:
                candidate = to_candidate(x)
                return _func(candidate)
            scipy_constraints.append({"type": "ineq", "fun": _constraint})

    result = minimize(
        wrapped,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
    )

    best_vector = result.x if result.success else x0
    candidate = to_candidate(best_vector)

    if constraints:
        for func in constraints:
            if func(candidate) < 0:
                raise RuntimeError("SLSQP produced parameters violating constraints")

    score = objective(candidate)
    return SearchResult(parameters=candidate, score=score)
