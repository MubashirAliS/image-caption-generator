import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import caption

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.caption_service import CaptionService

    logger.info("Loading model artifacts...")
    try:
        app.state.caption_service = CaptionService()
        logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model artifacts")
        app.state.caption_service = None
    yield


app = FastAPI(title="Image Caption Generator", lifespan=lifespan)

allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(caption.router, prefix="/api")
