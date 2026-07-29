from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exception_handlers import (
    register_exception_handlers,
)
from src.api.routes.health import router as health_router
from src.api.routes.model import router as model_router
from src.api.routes.prediction import (
    router as prediction_router,
)
from src.inference_engine import InferenceEngine
from src.logger import get_logger
from src.api.middleware import register_request_middleware
from src.metrics import MetricsRegistry

logger = get_logger(__name__)


EngineFactory = Callable[[], InferenceEngine]


def default_engine_factory() -> InferenceEngine:
    """Load the production inference engine from saved artifacts."""
    return InferenceEngine.from_artifacts()


def create_app(
    *,
    engine_factory: EngineFactory = default_engine_factory,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        logger.info("Loading production inference engine.")

        app.state.inference_engine = engine_factory()
        app.state.metrics_registry = MetricsRegistry()

        logger.info("Production inference engine loaded.")

        yield

        logger.info("Shutting down SF crime classification API.")

    app = FastAPI(
        title="San Francisco Crime Classification API",
        description=(
            "Production inference service for the Version 2 "
            "XGBoost crime-classification pipeline."
        ),
        version="3.0.0",
        lifespan=lifespan,
    )
    register_request_middleware(app)
    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(model_router)
    app.include_router(prediction_router)

    return app


app = create_app()