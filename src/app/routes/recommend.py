from typing import List

from fastapi import APIRouter, Depends, HTTPException

from src.app.dependencies import get_recommender
from src.app.schemas import GameResponse, RecommendationRequest
from src.ml.recommender import BoardGameRecommender

router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=List[GameResponse])
async def get_recommendations(
    payload: RecommendationRequest, recommender: BoardGameRecommender = Depends(get_recommender)
):
    """Get board game recommendations based on a game ID."""
    try:
        results = recommender.recommend(
            game_id=str(payload.game_id), weights=payload.weights, top_k=payload.top_k
        )
        return [
            GameResponse(
                name=row["name"],
                match_score=row["final_score"],
                cosine_similarity=row["cosine_similarity"],
                year_similarity=row["normalized_year_similarity"],
                playing_time_similarity=row["normalized_playing_time_similarity"],
                avg_rating=row["normalized_avg_rating"],
                popularity=row["normalized_popularity"],
                mechanics=row["mechanics"],
                categories=row["categories"],
                bgg_link=row["bgg_link"],
                comment=row["comment"],
            )
            for row in results.to_dicts()
        ]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
