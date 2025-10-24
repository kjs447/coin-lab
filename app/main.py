from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .database import engine
from .models import Base
from .schemas import (
    DataIngestionResponse,
    ExperimentRequest,
    ExperimentResponse,
    OptimizationOutcomeSchema,
    ParameterSensitivity,
    SensitivityPoint,
    StrategyInfo,
    StrategyParameter,
    TimeRange,
)
from .services import DataIngestionService
from .services.experiments import ExperimentService
from .strategies import registry as strategy_registry

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Coin Lab", version="0.1.0")

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _strategy_infos() -> List[StrategyInfo]:
    infos: List[StrategyInfo] = []
    for spec in strategy_registry.all():
        params = [
            StrategyParameter(
                name=param.name,
                default=param.default,
                kind=param.kind,
                recommended_min=param.recommended_min,
                recommended_max=param.recommended_max,
                description=param.description,
            )
            for param in spec.parameters
        ]
        infos.append(
            StrategyInfo(
                identifier=spec.identifier,
                name=spec.name,
                description=spec.description,
                parameters=params,
            )
        )
    return infos


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "strategies": _strategy_infos()})


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "database": settings.database_url}


@app.get("/strategies", response_model=List[StrategyInfo])
def list_strategies() -> List[StrategyInfo]:
    return _strategy_infos()


@app.post("/ingest", response_model=DataIngestionResponse)
def ingest_data(payload: TimeRange) -> DataIngestionResponse:
    service = DataIngestionService()
    summary = service.ingest(payload)
    return summary.as_response()


@app.post("/experiment", response_model=ExperimentResponse)
def run_experiment(payload: ExperimentRequest) -> ExperimentResponse:
    service = DataIngestionService()

    overrides = {
        p.name: {"min": p.min, "max": p.max, "default": p.default}
        for p in payload.parameter_ranges
    }
    experiment_service = ExperimentService(parameter_overrides=overrides)

    try:
        outcomes = experiment_service.run(
            strategy_id=payload.strategy_id,
            market=payload.market,
            validation_interval=(payload.validation.start, payload.validation.end),
            test_interval=(payload.test.start, payload.test.end),
            window_hours=payload.window_hours,
            validation_window_count=payload.validation_windows,
            test_window_count=payload.test_windows,
            ingestion_service=service,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    outcome_schemas: List[OptimizationOutcomeSchema] = []
    for outcome in outcomes:
        sensitivity = [
            ParameterSensitivity(
                parameter=name,
                points=[SensitivityPoint(value=point[0], metric_value=point[1]) for point in points],
            )
            for name, points in outcome.sensitivity.items()
        ]
        outcome_schemas.append(
            OptimizationOutcomeSchema(
                metric=outcome.metric,
                goal=outcome.goal,
                parameters=outcome.parameters,
                validation_metrics=outcome.validation_metrics,
                test_metrics=outcome.test_metrics,
                sensitivity=sensitivity,
            )
        )

    return ExperimentResponse(strategy_id=payload.strategy_id, outcomes=outcome_schemas)
