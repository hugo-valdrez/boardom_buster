from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.app.dependencies import state
from src.app.routes import recommend_router

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the recommender on startup."""
    state.initialize()
    yield


app = FastAPI(title="Boardom Buster API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(recommend_router)
# app.include_router(ingest_router)


@app.get("/")
async def read_root():
    """Serve the index.html file when opening the root URL."""
    return FileResponse(BASE_DIR / "static" / "templates" / "index.html")


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
