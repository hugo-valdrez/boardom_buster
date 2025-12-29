"""Tests for src/app/schemas.py"""

import pytest
from pydantic import ValidationError

from src.app.schemas import GameResponse, RecommendationRequest


class TestRecommendationRequest:
    """Test suite for the RecommendationRequest schema."""

    def test_valid_request_minimal(self):
        """Test valid request with minimal required fields."""
        request = RecommendationRequest(game_id=123, top_k=5)

        assert request.game_id == 123
        assert request.weights is None
        assert request.top_k == 5

    def test_valid_request_with_weights(self):
        """Test valid request with custom weights."""
        weights = {
            "weight_cosine_similarity": 0.3,
            "weight_year_similarity": 0.2,
            "weight_playing_time_similarity": 0.1,
            "weight_rating": 0.25,
            "weight_popularity": 0.15,
        }
        request = RecommendationRequest(game_id=123, weights=weights, top_k=10)

        assert request.game_id == 123
        assert request.weights == weights
        assert request.top_k == 10

    def test_request_without_top_k(self):
        """Test request defaults when top_k is not provided."""
        request = RecommendationRequest(game_id=456)

        assert request.game_id == 456
        assert request.top_k is None

    def test_invalid_game_id_type(self):
        """Test validation error for invalid game_id type."""
        with pytest.raises(ValidationError):
            RecommendationRequest(game_id="not_an_int", top_k=5)

    def test_game_id_as_string_coerced(self):
        """Test that numeric strings are coerced to int."""
        request = RecommendationRequest(game_id="123", top_k=5)
        assert request.game_id == 123


class TestGameResponse:
    """Test suite for the GameResponse schema."""

    def test_valid_response(self):
        """Test valid GameResponse creation."""
        response = GameResponse(
            name="Catan",
            match_score=0.95,
            cosine_similarity=0.9,
            year_similarity=0.85,
            playing_time_similarity=0.88,
            avg_rating=7.5,
            popularity=0.92,
            mechanics=["Dice Rolling", "Trading"],
            categories=["Family", "Strategy"],
            bgg_link="https://boardgamegeek.com/boardgame/13",
            comment="This game is very similar to your game!",
        )

        assert response.name == "Catan"
        assert response.match_score == 0.95
        assert response.cosine_similarity == 0.9
        assert response.year_similarity == 0.85
        assert response.playing_time_similarity == 0.88
        assert response.avg_rating == 7.5
        assert response.popularity == 0.92
        assert response.mechanics == ["Dice Rolling", "Trading"]
        assert response.categories == ["Family", "Strategy"]
        assert response.bgg_link == "https://boardgamegeek.com/boardgame/13"
        assert response.comment == "This game is very similar to your game!"

    def test_response_with_empty_lists(self):
        """Test GameResponse with empty mechanics and categories."""
        response = GameResponse(
            name="Test Game",
            match_score=0.5,
            cosine_similarity=0.5,
            year_similarity=0.5,
            playing_time_similarity=0.5,
            avg_rating=6.0,
            popularity=0.5,
            mechanics=[],
            categories=[],
            bgg_link="https://boardgamegeek.com/boardgame/1",
            comment="",
        )

        assert response.mechanics == []
        assert response.categories == []

    def test_invalid_response_missing_fields(self):
        """Test validation error when required fields are missing."""
        with pytest.raises(ValidationError):
            GameResponse(
                name="Incomplete Game",
                match_score=0.8,
                # Missing other required fields
            )

    def test_invalid_response_wrong_types(self):
        """Test validation error for incorrect field types."""
        with pytest.raises(ValidationError):
            GameResponse(
                name="Test Game",
                match_score="not_a_float",  # Should be float
                cosine_similarity=0.5,
                year_similarity=0.5,
                playing_time_similarity=0.5,
                avg_rating=6.0,
                popularity=0.5,
                mechanics=["Mechanic"],
                categories=["Category"],
                bgg_link="https://example.com",
                comment="",
            )
