"""FastAPI application for TempusSensor — cloud-ready."""

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.db import init_tables

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_tables()
    logger.info("database_ready")
    yield


app = FastAPI(
    title="TempusSensor API",
    description="Autonomous AI agent for DTC sensor optimization in biomagnetic anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tempussensor"}


@app.get("/")
async def root():
    return {
        "service": "TempusSensor",
        "version": "1.0.0",
        "description": "DTC Sensor Optimization Agent",
        "docs": "/docs",
        "api": "/api/v1",
    }
