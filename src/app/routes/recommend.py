from typing import List

from fastapi import APIRouter, Depends, HTTPException

from src.app.schemas import GameResponse, RecommendationRequest
from src.app.dependencies import get_recommender
from src.ml.recommender import BoardGameRecommender


router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=List[GameResponse])
async def get_recommendations(
    payload: RecommendationRequest,
    recommender: BoardGameRecommender = Depends(get_recommender)
):
    """Get board game recommendations based on a game ID."""    
    try:
        results = recommender.recommend(
            game_id=str(payload.game_id),
            weights=payload.weights,
            top_k=payload.top_k
        )
        return [
            GameResponse(
                id=int(row["id"]),
                name=row["name"],
                match_score=row["final_score"]
            )
            for row in results.to_dicts()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
