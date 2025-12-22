from pydantic import BaseModel
from typing import Optional

class RecommendationRequest(BaseModel):
    game_id: int
    weights: Optional[dict] = None
    top_k: int = None

class GameResponse(BaseModel):
    id: int
    name: str
    match_score: float