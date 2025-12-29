from pydantic import BaseModel
from typing import Optional

class RecommendationRequest(BaseModel):
    game_id: int
    weights: Optional[dict] = None
    top_k: int = None

class GameResponse(BaseModel):
    name: str
    match_score: float
    cosine_similarity: float
    year_similarity: float
    playing_time_similarity: float
    avg_rating: float
    popularity: float
    mechanics: list[str]
    categories: list[str]
    bgg_link: str
    comment: str