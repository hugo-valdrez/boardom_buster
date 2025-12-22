from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from src.app.dependencies import state
from src.app.routes import recommend_router, ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the recommender on startup."""
    state.initialize()
    yield


app = FastAPI(title="Boardom Buster API", lifespan=lifespan)

app.include_router(recommend_router)
app.include_router(ingest_router)


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)