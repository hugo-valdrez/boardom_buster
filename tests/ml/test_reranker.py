"""Tests for src/ml/reranker.py"""

import polars as pl
import pytest

from src.ml.reranker import Columns, ReRanker, ReRankerConfig


@pytest.fixture
def sample_input_game():
    """Fixture for a sample input game."""
    return pl.DataFrame(
        {
            "id": ["1"],
            "name": ["Input Game"],
            "publication_year": [2018],
            "playing_time": [60],
            "bayesian_avg_rating": [7.5],
            "avg_rating": [7.3],
            "popularity_score": [0.8],
        }
    )


@pytest.fixture
def sample_candidates():
    """Fixture for sample candidate games."""
    return pl.DataFrame(
        {
            "id": ["2", "3", "4"],
            "name": ["Candidate 1", "Candidate 2", "Candidate 3"],
            "cosine_distance": [0.1, 0.2, 0.3],
            "publication_year": [2017, 2019, 2020],
            "playing_time": [55, 70, 45],
            "bayesian_avg_rating": [7.8, 7.2, 6.9],
            "avg_rating": [7.6, 7.0, 6.7],
            "popularity_score": [0.85, 0.75, 0.70],
        }
    )


class TestReRankerConfig:
    """Test suite for the ReRankerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReRankerConfig()

        assert config.weight_cosine_similarity >= 0
        assert config.weight_year_similarity >= 0
        assert config.weight_playing_time_similarity >= 0
        assert config.weight_rating >= 0
        assert config.weight_popularity >= 0
        assert config.top_k >= 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReRankerConfig(
            weight_cosine_similarity=0.4,
            weight_year_similarity=0.1,
            weight_playing_time_similarity=0.1,
            weight_rating=0.3,
            weight_popularity=0.1,
            top_k=10,
        )

        assert config.weight_cosine_similarity == 0.4
        assert config.weight_year_similarity == 0.1
        assert config.top_k == 10

    def test_negative_weights_error(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            ReRankerConfig(weight_cosine_similarity=-0.1)

    def test_invalid_top_k(self):
        """Test that top_k < 1 raises error."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            ReRankerConfig(top_k=0)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "weight_cosine_similarity": 0.35,
            "weight_year_similarity": 0.15,
            "weight_playing_time_similarity": 0.1,
            "weight_rating": 0.25,
            "weight_popularity": 0.15,
            "top_k": 8,
        }
        config = ReRankerConfig.from_dict(config_dict)

        assert config.weight_cosine_similarity == 0.35
        assert config.top_k == 8

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ReRankerConfig(weight_cosine_similarity=0.4, weight_year_similarity=0.2, top_k=7)
        config_dict = config.to_dict()

        assert config_dict["weight_cosine_similarity"] == 0.4
        assert config_dict["weight_year_similarity"] == 0.2
        assert config_dict["top_k"] == 7


class TestColumns:
    """Test suite for the Columns constants class."""

    def test_column_constants_exist(self):
        """Test that all column name constants are defined."""
        assert hasattr(Columns, "GAME_ID")
        assert hasattr(Columns, "COSINE_DISTANCE")
        assert hasattr(Columns, "PUBLICATION_YEAR")
        assert hasattr(Columns, "PLAYING_TIME")
        assert hasattr(Columns, "BAYESIAN_RATING")
        assert hasattr(Columns, "AVG_RATING")
        assert hasattr(Columns, "POPULARITY_SCORE")
        assert hasattr(Columns, "COSINE_SIMILARITY")
        assert hasattr(Columns, "FINAL_SCORE")


class TestReRanker:
    """Test suite for the ReRanker class."""

    def test_initialization(self):
        """Test ReRanker initialization."""
        reranker = ReRanker()

        assert reranker.config is not None
        assert isinstance(reranker.config, ReRankerConfig)

    def test_rerank_basic(self, sample_input_game, sample_candidates):
        """Test basic re-ranking functionality."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=2, weights=weights)

        assert result.height <= 2
        assert "final_score" in result.columns
        assert "cosine_similarity" in result.columns

    def test_rerank_adds_cosine_similarity(self, sample_input_game, sample_candidates):
        """Test that cosine_similarity column is added."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=3, weights=weights)

        assert Columns.COSINE_SIMILARITY in result.columns

        result_sims = result[Columns.COSINE_SIMILARITY].to_list()

        for rs in result_sims:
            assert 0 <= rs <= 1

    def test_rerank_normalizes_scores(self, sample_input_game, sample_candidates):
        """Test that all scores are normalized to [0, 1]."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=3, weights=weights)

        # Check that all normalized columns are in [0, 1]
        for col in [
            Columns.NORM_YEAR_SIMILARITY,
            Columns.NORM_PLAYING_TIME_SIMILARITY,
            Columns.NORM_AVG_RATING,
            Columns.NORM_POPULARITY,
        ]:
            if col in result.columns:
                values = result[col].to_list()
                assert all(0 <= v <= 1 for v in values if v is not None)

    def test_rerank_sorts_by_final_score(self, sample_input_game, sample_candidates):
        """Test that results are sorted by final_score descending."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=3, weights=weights)

        scores = result[Columns.FINAL_SCORE].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_k(self, sample_input_game, sample_candidates):
        """Test that only top_k results are returned."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=1, weights=weights)

        assert result.height == 1

    def test_rerank_invalid_input_game_rows(self, sample_candidates):
        """Test error when input_game has != 1 row."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        # Create invalid input with 2 rows
        invalid_input = pl.DataFrame(
            {
                "id": ["1", "2"],
                "publication_year": [2018, 2019],
                "playing_time": [60, 70],
                "bayesian_avg_rating": [7.5, 7.8],
                "avg_rating": [7.3, 7.6],
                "popularity_score": [0.8, 0.85],
            }
        )

        with pytest.raises(ValueError, match="must have exactly 1 row"):
            reranker.rerank(invalid_input, sample_candidates, top_k=2, weights=weights)

    def test_rerank_output_columns(self, sample_input_game, sample_candidates):
        """Test that output contains expected columns."""
        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=3, weights=weights)

        expected_cols = [
            Columns.GAME_ID,
            "name",
            Columns.FINAL_SCORE,
            Columns.COSINE_SIMILARITY,
            Columns.NORM_YEAR_SIMILARITY,
            Columns.NORM_PLAYING_TIME_SIMILARITY,
            Columns.NORM_AVG_RATING,
            Columns.NORM_POPULARITY,
        ]

        for col in expected_cols:
            assert col in result.columns

    def test_rerank_uses_bayesian_rating_when_available(self, sample_input_game, sample_candidates):
        """Test that bayesian rating is preferred over avg_rating."""
        reranker = ReRanker()
        weights = {
            "weight_cosine_similarity": 0,
            "weight_year_similarity": 0,
            "weight_playing_time_similarity": 0,
            "weight_rating": 1.0,
            "weight_popularity": 0,
        }

        result = reranker.rerank(sample_input_game, sample_candidates, top_k=3, weights=weights)

        # Should have normalized rating column
        assert Columns.NORM_AVG_RATING in result.columns

    def test_rerank_falls_back_to_avg_rating(self, sample_input_game):
        """Test fallback to avg_rating when bayesian is 0 or null."""
        candidates_no_bayesian = pl.DataFrame(
            {
                "id": ["2", "3"],
                "name": ["Candidate 1", "Candidate 2"],
                "cosine_distance": [0.1, 0.2],
                "publication_year": [2017, 2019],
                "playing_time": [55, 70],
                "bayesian_avg_rating": [0, 0],  # Zero bayesian ratings
                "avg_rating": [7.6, 7.0],
                "popularity_score": [0.85, 0.75],
            }
        )

        reranker = ReRanker()
        weights = reranker.config.to_dict()

        result = reranker.rerank(
            sample_input_game, candidates_no_bayesian, top_k=2, weights=weights
        )

        # Should still work with avg_rating
        assert result.height > 0

    def test_extract_features(self, sample_input_game):
        """Test feature extraction from input game."""
        reranker = ReRanker()
        features = reranker._extract_features(sample_input_game)

        assert Columns.PUBLICATION_YEAR in features
        assert Columns.PLAYING_TIME in features
        assert Columns.BAYESIAN_RATING in features
        assert Columns.POPULARITY_SCORE in features

    def test_compute_final_score_with_custom_weights(self, sample_input_game, sample_candidates):
        """Test that custom weights affect final score."""
        reranker = ReRanker()

        # All weight on cosine similarity
        weights_cosine = {
            "weight_cosine_similarity": 1.0,
            "weight_year_similarity": 0,
            "weight_playing_time_similarity": 0,
            "weight_rating": 0,
            "weight_popularity": 0,
        }

        result1 = reranker.rerank(
            sample_input_game, sample_candidates, top_k=3, weights=weights_cosine
        )

        # All weight on popularity
        weights_pop = {
            "weight_cosine_similarity": 0,
            "weight_year_similarity": 0,
            "weight_playing_time_similarity": 0,
            "weight_rating": 0,
            "weight_popularity": 1.0,
        }

        result2 = reranker.rerank(
            sample_input_game, sample_candidates, top_k=3, weights=weights_pop
        )

        # Results should be different with different weights
        assert not result1[Columns.FINAL_SCORE].equals(result2[Columns.FINAL_SCORE])
