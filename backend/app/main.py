from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.exports import router as export_router
from app.api.reports import router as reports_router
from app.services.bootstrap import ensure_bootstrap
from app.services.model_hub import MODEL_HUB


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_bootstrap()
    yield


app = FastAPI(
    title="UrbanSightAI Pilot Backend",
    version="0.1.0",
    description="Policy-first urban governance prototype for ward-level prioritization.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "urbansightai-backend",
        "models_loaded": MODEL_HUB.prioritizer is not None,
        "metrics": MODEL_HUB.metrics,
    }


app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(reports_router)
app.include_router(export_router)
