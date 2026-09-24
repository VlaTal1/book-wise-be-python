import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import reading_speed, status
from core.config import settings
from services.rabbitmq_consumer import rabbitmq_consumer
from services.stress import model_store as stress_model_store
from services.test_generator import process_test_generation_request
from services.vosk_service import load_model

# Config logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def message_handler(payload):
    logger.info(f"Gor request for test generation: {payload}")

    await process_test_generation_request(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Vosk model for reading-speed feature")
    load_model()

    logger.info("Loading stress-check model (stress-accuracy layer)")
    stress_model_store.load_model()

    logger.info("Starting consuming test generation requests")
    await rabbitmq_consumer.start_consuming(callback=message_handler)

    yield

    logger.info("Stopping consuming test generation requests")
    await rabbitmq_consumer.stop_consuming()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(reading_speed.router, prefix="/ws", tags=["reading-speed"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
