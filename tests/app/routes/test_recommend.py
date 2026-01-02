"""Tests for src/app/routes/recommend.py"""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.app.routes.recommend import get_recommendations, router
from src.app.schemas import GameResponse, RecommendationRequest, RecommendationResponse
from src.ml.recommender import BoardGameRecommender


@pytest.fixture
def mock_recommender():
    """Fixture for a mock recommender."""
    recommender = Mock(spec=BoardGameRecommender)
    # Setup _knn mock for get_input_game
    recommender._knn = Mock()
    recommender._knn.get_input_game = Mock(
        return_value=pl.DataFrame(
            {
                "id": ["123"],
                "name": ["Input Game"],
                "image_url": ["https://example.com/input.jpg"],
                "description": ["Input game description"],
                "categories": [["Strategy"]],
                "mechanics": [["Trading"]],
                "bgg_link": ["https://boardgamegeek.com/boardgame/123"],
            }
        )
    )
    return recommender


@pytest.fixture
def sample_recommendation_data():
    """Fixture for sample recommendation data."""
    return pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["Game 1", "Game 2", "Game 3"],
            "final_score": [0.95, 0.90, 0.85],
            "cosine_similarity": [0.92, 0.88, 0.83],
            "normalized_year_similarity": [0.90, 0.85, 0.80],
            "normalized_playing_time_similarity": [0.88, 0.87, 0.86],
            "normalized_avg_rating": [0.95, 0.90, 0.85],
            "normalized_popularity": [0.93, 0.89, 0.84],
            "mechanics": [["Trading"], ["Dice Rolling"], ["Area Control"]],
            "categories": [["Strategy"], ["Family"], ["War"]],
            "bgg_link": [
                "https://boardgamegeek.com/boardgame/1",
                "https://boardgamegeek.com/boardgame/2",
                "https://boardgamegeek.com/boardgame/3",
            ],
            "comment": ["Great game!", "Fun for families!", "Deep strategy!"],
            "image_url": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
                "https://example.com/image3.jpg",
            ],
            "description": ["Desc 1", "Desc 2", "Desc 3"],
        }
    )


class TestGetRecommendations:
    """Test suite for the get_recommendations endpoint."""

    def test_get_recommendations_success(self, mock_recommender, sample_recommendation_data):
        """Test successful recommendation retrieval."""
        import asyncio

        async def run_test():
            mock_recommender.recommend.return_value = sample_recommendation_data

            request = RecommendationRequest(game_id=123, top_k=3)
            result = await get_recommendations(request, mock_recommender)

            assert isinstance(result, RecommendationResponse)
            assert result.input_game.name == "Input Game"
            assert len(result.recommendations) == 3
            assert isinstance(result.recommendations[0], GameResponse)
            assert result.recommendations[0].name == "Game 1"
            assert result.recommendations[0].match_score == 0.95

            mock_recommender.recommend.assert_called_once_with(game_id="123", weights=None, top_k=3)

        asyncio.run(run_test())

    def test_get_recommendations_with_weights(self, mock_recommender, sample_recommendation_data):
        """Test recommendations with custom weights."""
        import asyncio

        async def run_test():
            mock_recommender.recommend.return_value = sample_recommendation_data

            weights = {
                "weight_cosine_similarity": 0.4,
                "weight_year_similarity": 0.2,
                "weight_playing_time_similarity": 0.1,
                "weight_rating": 0.2,
                "weight_popularity": 0.1,
            }
            request = RecommendationRequest(game_id=456, weights=weights, top_k=3)

            result = await get_recommendations(request, mock_recommender)

            assert isinstance(result, RecommendationResponse)
            assert len(result.recommendations) == 3
            mock_recommender.recommend.assert_called_once_with(
                game_id="456", weights=weights, top_k=3
            )

        asyncio.run(run_test())

    def test_get_recommendations_game_not_found(self, mock_recommender):
        """Test 404 error when game is not found."""
        import asyncio

        async def run_test():
            # Mock _knn.get_input_game to raise ValueError
            mock_recommender._knn.get_input_game.side_effect = ValueError("Game not found")

            request = RecommendationRequest(game_id=999, top_k=5)

            with pytest.raises(HTTPException) as exc_info:
                await get_recommendations(request, mock_recommender)

            assert exc_info.value.status_code == 404
            assert "Game not found" in str(exc_info.value.detail)

        asyncio.run(run_test())

    def test_get_recommendations_unexpected_error(self, mock_recommender):
        """Test 500 error on unexpected exception."""
        import asyncio

        async def run_test():
            mock_recommender._knn.get_input_game.side_effect = RuntimeError("Unexpected error")

            request = RecommendationRequest(game_id=123, top_k=5)

            with pytest.raises(HTTPException) as exc_info:
                await get_recommendations(request, mock_recommender)

            assert exc_info.value.status_code == 500
            assert "unexpected error" in str(exc_info.value.detail).lower()

        asyncio.run(run_test())

    def test_get_recommendations_response_structure(
        self, mock_recommender, sample_recommendation_data
    ):
        """Test that response matches RecommendationResponse schema."""
        import asyncio

        async def run_test():
            mock_recommender.recommend.return_value = sample_recommendation_data

            request = RecommendationRequest(game_id=123, top_k=3)
            result = await get_recommendations(request, mock_recommender)

            # Check response structure
            assert isinstance(result, RecommendationResponse)
            assert hasattr(result, "input_game")
            assert hasattr(result, "recommendations")

            # Check input_game structure
            assert hasattr(result.input_game, "id")
            assert hasattr(result.input_game, "name")
            assert hasattr(result.input_game, "image")
            assert hasattr(result.input_game, "description")

            # Check recommendations structure
            for game in result.recommendations:
                assert hasattr(game, "name")
                assert hasattr(game, "match_score")
                assert hasattr(game, "cosine_similarity")
                assert hasattr(game, "year_similarity")
                assert hasattr(game, "playing_time_similarity")
                assert hasattr(game, "avg_rating")
                assert hasattr(game, "popularity")
                assert hasattr(game, "mechanics")
                assert hasattr(game, "categories")
                assert hasattr(game, "bgg_link")
                assert hasattr(game, "comment")

        asyncio.run(run_test())
