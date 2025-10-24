from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import backtrader as bt
import pandas as pd
import numpy as np
from sqlalchemy import select

from ..config import settings
from ..database import session_scope
from ..metrics.base import BacktestStats
from ..metrics.evaluation import METRICS
from ..optimization.search import ConstraintFn, random_search, slsqp_optimize
from ..models import OHLCV
from ..strategies import registry as strategy_registry
from ..strategies.base import StrategySpec
from ..utils.time import to_naive_utc
from ..schemas import TimeRange
from .data_ingestion import DataIngestionService


@dataclass
class BacktestResult:
    stats: BacktestStats
    parameters: Dict[str, float]


@dataclass
class WindowSlice:
    data: pd.DataFrame
    start_idx: int
    end_idx: int
    full_df: pd.DataFrame

    @property
    def start_datetime(self) -> datetime | None:
        if self.data.empty:
            return None
        return self.data.index[0].to_pydatetime()

    @property
    def end_datetime(self) -> datetime | None:
        if self.data.empty:
            return None
        return self.data.index[-1].to_pydatetime()


class BacktestRunner:
    def __init__(self, cash: float = 100_000_000.0) -> None:
        self.cash = cash

    def run(
        self,
        df: pd.DataFrame,
        spec: StrategySpec,
        parameters: Dict[str, float],
        focus_start: datetime | None = None,
    ) -> BacktestResult:
        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(
            dataname=df,
            timeframe=bt.TimeFrame.Minutes,
            compression=60,
        )
        cerebro.adddata(data_feed)
        casted_params = {
            param.name: int(parameters.get(param.name, param.default))
            if param.kind == "int"
            else float(parameters.get(param.name, param.default))
            for param in spec.parameters
        }
        if focus_start is not None:
            casted_params["live_start"] = focus_start
        cerebro.addstrategy(spec.strategy_cls, **casted_params)
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=0.0005)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

        results = cerebro.run()
        strat = results[0]

        returns_map = strat.analyzers.timereturn.get_analysis()
        ordered_returns = sorted(returns_map.items(), key=lambda item: item[0])

        def _to_datetime(obj) -> datetime:
            try:
                return pd.Timestamp(obj).to_pydatetime()
            except Exception:  # noqa: BLE001
                return focus_start or datetime.min

        filtered_returns = [
            (dt, ret)
            for dt, ret in ordered_returns
            if focus_start is None or _to_datetime(dt) >= focus_start
        ]

        equity_curve = [self.cash]
        value = self.cash
        returns_series: List[float] = []
        for _, r in filtered_returns:
            value *= 1 + r
            equity_curve.append(value)
            returns_series.append(r)
        final_value = strat.broker.getvalue()
        if equity_curve:
            equity_curve[-1] = final_value
        else:
            equity_curve = [self.cash, final_value]
            returns_series = [final_value / self.cash - 1]

        trade_analysis = strat.analyzers.trade.get_analysis() or {}
        closed_trades = trade_analysis.get("closed", [])
        if isinstance(closed_trades, dict):
            closed_trades = closed_trades.values()
        filtered_trades = []
        for trade in closed_trades:
            if not isinstance(trade, dict):
                continue
            open_dt = trade.get("dtopen")
            open_dt = pd.Timestamp(open_dt).to_pydatetime() if open_dt is not None else None
            if focus_start is None or (open_dt is not None and open_dt >= focus_start):
                filtered_trades.append(trade)
        winning_trades = sum(1 for trade in filtered_trades if trade.get("pnl", 0) > 0)
        losing_trades = sum(1 for trade in filtered_trades if trade.get("pnl", 0) <= 0)
        closed = len(filtered_trades)

        drawdown_analysis = strat.analyzers.drawdown.get_analysis() or {}
        max_dd = drawdown_analysis.get("max", {})
        max_drawdown = float(max_dd.get("drawdown", 0.0)) / 100.0
        max_drawdown_money = float(max_dd.get("moneydown", 0.0))

        stats = BacktestStats(
            returns=list(returns_series),
            equity_curve=equity_curve,
            max_drawdown=max_drawdown,
            max_drawdown_money=max_drawdown_money,
            total_trades=int(closed),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            final_value=final_value,
            initial_value=self.cash,
        )
        return BacktestResult(stats=stats, parameters=casted_params)


def fetch_dataframe(market: str, start: datetime, end: datetime) -> pd.DataFrame:
    start = to_naive_utc(start)
    end = to_naive_utc(end)
    with session_scope() as session:
        stmt = (
            select(OHLCV)
            .where(OHLCV.market == market)
            .where(OHLCV.timestamp >= start)
            .where(OHLCV.timestamp <= end)
            .order_by(OHLCV.timestamp.asc())
        )
        rows = list(session.scalars(stmt))
    if not rows:
        raise RuntimeError("Requested data range is empty")
    data = {
        "datetime": [row.timestamp for row in rows],
        "open": [row.opening_price for row in rows],
        "high": [row.high_price for row in rows],
        "low": [row.low_price for row in rows],
        "close": [row.trade_price for row in rows],
        "volume": [row.candle_acc_trade_volume for row in rows],
    }
    df = pd.DataFrame(data).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df


def split_into_windows(
    df: pd.DataFrame,
    window_hours: int,
    count: int,
    seed: int | None = None,
    full_df: pd.DataFrame | None = None,
    min_start_index: int = 0,
) -> List[WindowSlice]:
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if len(df) < window_hours:
        raise ValueError("Dataframe shorter than window length")

    rng = random.Random(seed or settings.random_seed)
    windows: List[WindowSlice] = []
    available_indices = list(range(0, len(df) - window_hours + 1))
    available_indices = [idx for idx in available_indices if idx >= min_start_index]
    rng.shuffle(available_indices)

    used_indices: set[int] = set()
    for start_idx in available_indices:
        if len(windows) >= count:
            break
        overlap = any(idx in used_indices for idx in range(start_idx, start_idx + window_hours))
        if overlap:
            continue
        window_df = df.iloc[start_idx : start_idx + window_hours].copy()
        windows.append(
            WindowSlice(
                data=window_df,
                start_idx=start_idx,
                end_idx=start_idx + window_hours,
                full_df=full_df if full_df is not None else df,
            )
        )
        used_indices.update(range(start_idx, start_idx + window_hours))

    if len(windows) < count:
        raise RuntimeError("Unable to sample requested number of windows without overlap")
    return windows


class StrategyEvaluator:
    def __init__(self) -> None:
        self.runner = BacktestRunner()

    def _minimum_required_bars(self, spec: StrategySpec, parameters: Dict[str, float]) -> int:
        required = 1
        for param in spec.parameters:
            if "period" in param.name.lower():
                value = parameters.get(param.name, param.default)
                try:
                    period_value = int(round(float(value)))
                except (TypeError, ValueError):
                    continue
                required = max(required, period_value)
        return max(required, 1)

    def evaluate(
        self,
        spec: StrategySpec,
        parameters: Dict[str, float],
        windows: Sequence[WindowSlice],
    ) -> Dict[str, List[float]]:
        metric_values: Dict[str, List[float]] = {name: [] for name in METRICS.keys()}
        for entry in windows:
            window_slice = (
                entry
                if isinstance(entry, WindowSlice)
                else WindowSlice(
                    data=entry.copy(),
                    start_idx=0,
                    end_idx=len(entry),
                    full_df=entry,
                )
            )
            window = window_slice.data
            focus_start = window.index[0] if not window.empty else None
            min_bars = self._minimum_required_bars(spec, parameters)

            warmup_start_idx = max(0, window_slice.start_idx - min_bars)
            extended = window_slice.full_df.iloc[warmup_start_idx : window_slice.end_idx].copy()

            if len(extended) < min_bars:
                start = window_slice.start_datetime or "unknown"
                end = window_slice.end_datetime or "unknown"
                raise RuntimeError(
                    f"Window {start} -> {end} has only {len(extended)} candles available (including warmup) "
                    f"but requires at least {min_bars}. Consider enlarging the data range or reducing strategy periods."
                )

            focus_dt = focus_start.to_pydatetime() if focus_start is not None else None
            try:
                result = self.runner.run(extended, spec, parameters, focus_start=focus_dt)
            except Exception as exc:  # noqa: BLE001
                start = window_slice.start_datetime or "unknown"
                end = window_slice.end_datetime or "unknown"
                raise RuntimeError(
                    f"Backtest failed for window {start} -> {end} with parameters {parameters}: {exc}"
                ) from exc
            stats = result.stats
            for name, metric in METRICS.items():
                metric_values[name].append(metric.compute(stats))
        return metric_values


def aggregate_metric_values(metric_values: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not metric_values:
        return summary
    cumulative = metric_values.get('cumulative_return', [])
    volatility = metric_values.get('volatility', [])
    drawdown = metric_values.get('max_drawdown', [])
    sharpe = metric_values.get('sharpe_ratio', [])
    win_rate = metric_values.get('win_rate', [])

    if cumulative:
        summary['cumulative_return'] = float(pd.Series(cumulative).median())
    if volatility:
        summary['volatility'] = float(pd.Series(volatility).mean())
    if drawdown:
        dd_series = pd.Series(drawdown)
        summary['max_drawdown_mean'] = float(dd_series.mean())
        summary['max_drawdown_max'] = float(dd_series.max())
    if sharpe:
        summary['sharpe_ratio'] = float(pd.Series(sharpe).mean())
    if win_rate:
        summary['win_rate'] = float(pd.Series(win_rate).mean())
    return summary


@dataclass
class OptimizationOutcome:
    metric: str
    goal: str
    parameters: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    sensitivity: Dict[str, List[Tuple[float, float]]]  # parameter -> list of (value, metric)


OBJECTIVES = [
    {"metric": "cumulative_return", "goal": "max"},
    {"metric": "volatility", "goal": "min"},
    {"metric": "max_drawdown_mean", "goal": "min"},
    {"metric": "max_drawdown_max", "goal": "min"},
    {"metric": "sharpe_ratio", "goal": "max"},
    {"metric": "win_rate", "goal": "max"},
]


class ExperimentService:
    SENSITIVITY_POINTS = 10

    def __init__(self, parameter_overrides: Dict[str, Dict[str, float]] | None = None) -> None:
        self.evaluator = StrategyEvaluator()
        self.parameter_overrides = parameter_overrides or {}

    def _warmup_hours(self, parameters_meta) -> int:
        max_hours = 0
        for meta in parameters_meta:
            if "period" in meta.name.lower() or meta.kind == "int":
                candidates = [meta.default]
                if meta.recommended_max is not None:
                    candidates.append(meta.recommended_max)
                if meta.recommended_min is not None:
                    candidates.append(meta.recommended_min)
                numeric_candidates = [c for c in candidates if c is not None]
                if not numeric_candidates:
                    continue
                max_candidate = max(abs(float(c)) for c in numeric_candidates)
                max_hours = max(max_hours, int(round(max_candidate)))
        return max(max_hours, 0)

    def _build_parameters(self, spec: StrategySpec) -> List:
        params = []
        for param in spec.parameters:
            override = self.parameter_overrides.get(param.name, {})
            params.append(type(param)(
                name=param.name,
                default=override.get('default', param.default),
                kind=param.kind,
                recommended_min=override.get('min', param.recommended_min),
                recommended_max=override.get('max', param.recommended_max),
                description=param.description,
            ))
        return params

    def _build_constraints(self, spec: StrategySpec) -> List[ConstraintFn]:
        constraints: List[ConstraintFn] = []
        identifier = spec.identifier
        if identifier == "ma_cross":
            constraints.append(lambda p: p["long_period"] - p["short_period"] - 1.0)
        if identifier == "obv_ma":
            constraints.append(lambda p: p["long_period"] - p["short_period"] - 1.0)
        if identifier == "bollinger_rsi":
            constraints.append(lambda p: p["rsi_high"] - p["rsi_low"] - 5.0)
        return constraints

    def _objective_value(
        self,
        spec: StrategySpec,
        parameters: Dict[str, float],
        windows: Sequence[WindowSlice],
        metric_key: str,
        goal: str,
    ) -> float:
        metrics = self.evaluator.evaluate(spec, parameters, windows)
        summary = aggregate_metric_values(metrics)
        if metric_key not in summary:
            return float('-inf') if goal == 'max' else float('inf')
        return summary[metric_key]

    def _optimize_metric(
        self,
        spec: StrategySpec,
        parameters_meta,
        validation_windows: Sequence[WindowSlice],
        metric: str,
        goal: str,
        constraints: Sequence[ConstraintFn],
    ) -> Dict[str, float]:
        def objective(candidate: Dict[str, float]) -> float:
            return self._objective_value(spec, candidate, validation_windows, metric, goal)

        random_best = random_search(
            parameters_meta,
            settings.random_search_trials,
            objective,
            goal,
            constraints=constraints,
        )

        try:
            slsqp_best = slsqp_optimize(
                random_best.parameters,
                parameters_meta,
                objective,
                goal,
                constraints=constraints,
            )
        except RuntimeError:
            slsqp_best = random_best

        best = random_best
        if (goal == 'max' and slsqp_best.score > random_best.score) or (
            goal == 'min' and slsqp_best.score < random_best.score
        ):
            best = slsqp_best
        return best.parameters

    def _sensitivity(
        self,
        spec: StrategySpec,
        parameters_meta,
        best_params: Dict[str, float],
        metric: str,
        test_windows: Sequence[WindowSlice],
    ) -> Dict[str, List[Tuple[float, float]]]:
        curves: Dict[str, List[Tuple[float, float]]] = {}
        for meta in parameters_meta:
            lower = meta.recommended_min if meta.recommended_min is not None else meta.default
            upper = meta.recommended_max if meta.recommended_max is not None else meta.default
            values: List[float]
            if meta.kind == 'int':
                lower_i = int(round(lower))
                upper_i = int(round(upper))
                if upper_i < lower_i:
                    lower_i, upper_i = upper_i, lower_i
                step = max(1, (upper_i - lower_i) // max(1, self.SENSITIVITY_POINTS - 1))
                values = list(range(lower_i, upper_i + 1, step))
                if values[-1] != upper_i:
                    values.append(upper_i)
            else:
                values = list(np.linspace(lower, upper, self.SENSITIVITY_POINTS))
            points: List[Tuple[float, float]] = []
            for value in values:
                candidate = dict(best_params)
                candidate[meta.name] = int(round(value)) if meta.kind == 'int' else float(value)
                metrics = self.evaluator.evaluate(spec, candidate, test_windows)
                summary = aggregate_metric_values(metrics)
                if metric in summary:
                    points.append((float(candidate[meta.name]), summary[metric]))
            curves[meta.name] = points
        return curves

    def run(
        self,
        strategy_id: str,
        market: str,
        validation_interval: tuple[datetime, datetime],
        test_interval: tuple[datetime, datetime],
        window_hours: int,
        validation_window_count: int,
        test_window_count: int,
        ingestion_service: DataIngestionService | None = None,
    ) -> List[OptimizationOutcome]:
        spec = strategy_registry.get(strategy_id)
        parameters_meta = self._build_parameters(spec)
        constraints = self._build_constraints(spec)
        warmup_hours = self._warmup_hours(parameters_meta)

        ingest = ingestion_service or DataIngestionService()

        validation_start, validation_end = validation_interval
        test_start, test_end = test_interval

        ingest.ingest(TimeRange(market=market, start=validation_start, end=validation_end))
        ingest.ingest(TimeRange(market=market, start=test_start, end=test_end))

        validation_df = fetch_dataframe(market, validation_start, validation_end)
        test_df = fetch_dataframe(market, test_start, test_end)

        min_start_index = warmup_hours if warmup_hours > 0 else 0

        validation_windows = split_into_windows(
            validation_df,
            window_hours,
            validation_window_count,
            full_df=validation_df,
            min_start_index=min_start_index,
        )
        test_windows = split_into_windows(
            test_df,
            window_hours,
            test_window_count,
            full_df=test_df,
            min_start_index=min_start_index,
        )
        outcomes: List[OptimizationOutcome] = []
        for objective in OBJECTIVES:
            metric_key = objective['metric']
            goal = objective['goal']
            best_params = self._optimize_metric(
                spec,
                parameters_meta,
                validation_windows,
                metric_key,
                goal,
                constraints,
            )
            validation_metrics_values = self.evaluator.evaluate(spec, best_params, validation_windows)
            validation_summary = aggregate_metric_values(validation_metrics_values)
            test_metrics_values = self.evaluator.evaluate(spec, best_params, test_windows)
            test_summary = aggregate_metric_values(test_metrics_values)
            sensitivity = self._sensitivity(spec, parameters_meta, best_params, metric_key, test_windows)
            outcomes.append(
                OptimizationOutcome(
                    metric=metric_key,
                    goal=goal,
                    parameters=best_params,
                    validation_metrics=validation_summary,
                    test_metrics=test_summary,
                    sensitivity=sensitivity,
                )
            )
        return outcomes
