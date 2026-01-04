from fastapi import APIRouter, Depends, HTTPException

from src.app.dependencies import get_recommender
from src.app.schemas import (
    GameResponse,
    InputGameInfo,
    RecommendationRequest,
    RecommendationResponse,
)
from src.ml.recommender import BoardGameRecommender

router = APIRouter(tags=["recommendations"])


@router.get("/games")
async def get_all_games(recommender: BoardGameRecommender = Depends(get_recommender)):
    """Get all available games for client-side filtering."""
    try:
        games_df = recommender._knn._full_df
        results = games_df.select(["id", "name", "thumbnail_url"]).to_dicts()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch games: {str(e)}")


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    payload: RecommendationRequest, recommender: BoardGameRecommender = Depends(get_recommender)
):
    """Get board game recommendations based on a game ID."""
    try:
        # Get input game info
        input_game_df = recommender._knn.get_input_game(str(payload.game_id))
        input_game_row = input_game_df.to_dicts()[0]
        input_game = InputGameInfo(
            id=str(input_game_row["id"]),
            name=input_game_row["name"],
            image=input_game_row.get("image_url"),
            description=input_game_row.get("description"),
            categories=input_game_row.get("categories"),
            mechanics=input_game_row.get("mechanics"),
            bgg_link=input_game_row.get("bgg_link"),
        )

        # Get recommendations
        results = recommender.recommend(
            game_id=str(payload.game_id),
            weights=payload.weights,
            top_k=payload.top_k,
            exclude_same_family=payload.exclude_same_family,
        )
        recommendations = [
            GameResponse(
                name=row["name"],
                match_score=row["final_score"],
                cosine_similarity=row["cosine_similarity"],
                difficulty_similarity=row["normalized_difficulty_similarity"],
                playing_time_similarity=row["normalized_playing_time_similarity"],
                avg_rating=row["normalized_avg_rating"],
                popularity=row["normalized_popularity"],
                mechanics=row["mechanics"],
                categories=row["categories"],
                bgg_link=row["bgg_link"],
                comment=row["comment"],
                image=row["image_url"],
                description=row.get("description"),
            )
            for row in results.to_dicts()
        ]

        return RecommendationResponse(input_game=input_game, recommendations=recommendations)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
