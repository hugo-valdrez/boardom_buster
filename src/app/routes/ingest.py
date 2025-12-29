import asyncio

from fastapi import APIRouter, BackgroundTasks

from src.app.dependencies import state
from src.etl.ingestion.bgg import continuous_fetch
from src.etl.consolidation.consolidation import consolidation


router = APIRouter(tags=["ingestion"])


async def run_ingestion_pipeline():
    """Run the full ingestion and consolidation pipeline, then reload recommender."""
    # Run ingestion
    await continuous_fetch()
    
    # Run consolidation
    consolidation()
    
    # Reload recommender with new data
    state.reload()


@router.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Trigger the ingestion pipeline in the background."""
    background_tasks.add_task(asyncio.create_task, run_ingestion_pipeline())
    return {"status": "Ingestion pipeline started in background"}
