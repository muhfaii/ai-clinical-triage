"""
Clinical AI API — FastAPI service wrapping all 4 ML models.

Usage:
    cd /Users/loop/Documents/ai-agent-hospital
    source .venv/bin/activate
    uvicorn clinical_api.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import ards, chat, diabetes, health, heart_attack, sit2stand, stroke, triage
from .services import model_store
from triage_chat.agent import triage_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load_all()
    await triage_agent.start()
    yield
    await triage_agent.stop()


app = FastAPI(
    title="Clinical AI API",
    description="Decision support predictions for stroke, heart attack, diabetes, and ARDS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(stroke.router, prefix="/api/v1", tags=["stroke"])
app.include_router(heart_attack.router, prefix="/api/v1", tags=["heart_attack"])
app.include_router(diabetes.router, prefix="/api/v1", tags=["diabetes"])
app.include_router(ards.router, prefix="/api/v1", tags=["ards"])
app.include_router(triage.router, prefix="/api/v1", tags=["triage"])
app.include_router(sit2stand.router, prefix="/api/v1", tags=["sit2stand"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
