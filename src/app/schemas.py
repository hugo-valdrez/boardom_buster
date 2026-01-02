from typing import Optional

from pydantic import BaseModel


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
    image: Optional[str] = None
    description: Optional[str] = None


class InputGameInfo(BaseModel):
    id: str
    name: str
    image: Optional[str] = None
    description: Optional[str] = None
    categories: Optional[list[str]] = None
    mechanics: Optional[list[str]] = None
    bgg_link: Optional[str] = None


class RecommendationResponse(BaseModel):
    input_game: InputGameInfo
    recommendations: list[GameResponse]
