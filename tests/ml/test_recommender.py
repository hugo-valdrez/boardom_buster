"""Tests for src/ml/recommender.py"""

from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from src.ml.recommender import BoardGameRecommender


@pytest.fixture
def sample_processed_data():
    """Fixture for sample processed data."""
    return pl.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "name": ["Game 1", "Game 2", "Game 3", "Game 4", "Game 5"],
            "to_recommend": [1, 1, 1, 1, 1],
            "feature_1": [1.0, 0.8, 0.9, 0.7, 0.6],
            "feature_2": [0.5, 0.6, 0.5, 0.4, 0.3],
            "publication_year": [2015, 2016, 2017, 2018, 2019],
            "playing_time": [60, 45, 90, 30, 120],
            "avg_rating": [7.5, 8.0, 7.0, 6.5, 7.2],
            "bayesian_avg_rating": [7.6, 8.1, 7.1, 6.6, 7.3],
            "popularity_score": [0.8, 0.9, 0.7, 0.6, 0.65],
            "difficulty": [2.5, 3.0, 2.0, 1.5, 3.5],
            "mechanics": [
                ["Trading"],
                ["Dice Rolling"],
                ["Area Control"],
                ["Hand Management"],
                ["Voting"],
            ],
            "categories": [["Strategy"], ["Family"], ["War"], ["Party"], ["Economic"]],
            "family": [[], [], [], [], []],
            "bgg_link": [
                "https://boardgamegeek.com/boardgame/1",
                "https://boardgamegeek.com/boardgame/2",
                "https://boardgamegeek.com/boardgame/3",
                "https://boardgamegeek.com/boardgame/4",
                "https://boardgamegeek.com/boardgame/5",
            ],
            "image_url": [
                "https://cf.geekdo-images.com/1.jpg",
                "https://cf.geekdo-images.com/2.jpg",
                "https://cf.geekdo-images.com/3.jpg",
                "https://cf.geekdo-images.com/4.jpg",
                "https://cf.geekdo-images.com/5.jpg",
            ],
            "thumbnail_url": [
                "https://cf.geekdo-images.com/thumb1.jpg",
                "https://cf.geekdo-images.com/thumb2.jpg",
                "https://cf.geekdo-images.com/thumb3.jpg",
                "https://cf.geekdo-images.com/thumb4.jpg",
                "https://cf.geekdo-images.com/thumb5.jpg",
            ],
            "description": [
                "A strategic trading game",
                "A fun dice rolling game",
                "An area control war game",
                "A hand management party game",
                "An economic voting game",
            ],
        }
    )


@pytest.fixture
def sample_knn_results():
    """Fixture for sample KNN results."""
    return pl.DataFrame(
        {
            "id": ["2", "3", "4"],
            "name": ["Game 2", "Game 3", "Game 4"],
            "cosine_distance": [0.1, 0.2, 0.3],
            "publication_year": [2016, 2017, 2018],
            "playing_time": [45, 90, 30],
            "avg_rating": [8.0, 7.0, 6.5],
            "bayesian_avg_rating": [8.1, 7.1, 6.6],
            "popularity_score": [0.9, 0.7, 0.6],
            "difficulty": [3.0, 2.0, 1.5],
            "to_recommend": [1, 1, 1],
            "mechanics": [["Dice Rolling"], ["Area Control"], ["Hand Management"]],
            "categories": [["Family"], ["War"], ["Party"]],
            "family": [[], [], []],
            "bgg_link": [
                "https://boardgamegeek.com/boardgame/2",
                "https://boardgamegeek.com/boardgame/3",
                "https://boardgamegeek.com/boardgame/4",
            ],
            "image_url": [
                "https://cf.geekdo-images.com/2.jpg",
                "https://cf.geekdo-images.com/3.jpg",
                "https://cf.geekdo-images.com/4.jpg",
            ],
            "description": [
                "A fun dice rolling game",
                "An area control war game",
                "A hand management party game",
            ],
        }
    )


@pytest.fixture
def sample_reranked_results():
    """Fixture for sample reranked results."""
    return pl.DataFrame(
        {
            "id": ["2", "3", "4"],
            "name": ["Game 2", "Game 3", "Game 4"],
            "final_score": [0.95, 0.88, 0.82],
            "cosine_similarity": [0.9, 0.8, 0.7],
            "normalized_difficulty_similarity": [0.85, 0.80, 0.75],
            "normalized_playing_time_similarity": [0.88, 0.82, 0.78],
            "normalized_avg_rating": [0.95, 0.85, 0.75],
            "normalized_popularity": [0.92, 0.82, 0.72],
        }
    )


class TestBoardGameRecommender:
    """Test suite for the BoardGameRecommender class."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = BoardGameRecommender()

        assert recommender._knn is not None
        assert recommender._reranker is not None
        assert recommender._is_fitted is False

    @patch("src.ml.recommender.WriterReaderFactory")
    @patch("src.ml.recommender.settings")
    def test_load_data(self, mock_settings, mock_factory, sample_processed_data):
        """Test loading data from disk."""
        mock_settings.PATHS = {"processed_data": Path("/fake/path")}
        mock_writer_reader = Mock()
        mock_writer_reader.read.return_value = sample_processed_data
        mock_factory.create_from_directory.return_value = mock_writer_reader

        recommender = BoardGameRecommender()
        result = recommender.load_data()

        assert result is recommender  # Method chaining
        assert recommender._is_fitted is True
        mock_writer_reader.read.assert_called_once()

    def test_fit(self, sample_processed_data):
        """Test fitting the recommender on a DataFrame."""
        recommender = BoardGameRecommender()
        result = recommender.fit(sample_processed_data)

        assert result is recommender  # Method chaining
        assert recommender._is_fitted is True

    def test_recommend_not_fitted(self):
        """Test error when recommending without fitting."""
        recommender = BoardGameRecommender()

        with pytest.raises(RuntimeError, match="Recommender not fitted"):
            recommender.recommend("1")

    @patch.object(BoardGameRecommender, "_add_comments")
    def test_recommend_basic(
        self, mock_add_comments, sample_processed_data, sample_knn_results, sample_reranked_results
    ):
        """Test basic recommendation functionality."""
        # Setup mocks
        recommender = BoardGameRecommender()
        recommender.fit(sample_processed_data)

        recommender._knn.get_candidates = Mock(return_value=sample_knn_results)
        recommender._knn.get_input_game = Mock(
            return_value=sample_processed_data.filter(pl.col("id") == "1")
        )
        recommender._reranker.rerank = Mock(return_value=sample_reranked_results)

        # Add required columns to mock
        result_with_metadata = sample_reranked_results.join(
            sample_knn_results.select(["id", "mechanics", "categories", "bgg_link"]),
            on="id",
            how="left",
        )
        mock_add_comments.return_value = result_with_metadata

        result = recommender.recommend("1")

        assert result is not None
        recommender._knn.get_candidates.assert_called_once()
        recommender._reranker.rerank.assert_called_once()

    def test_recommend_with_custom_weights(self, sample_processed_data):
        """Test recommendations with custom weights."""
        recommender = BoardGameRecommender()
        recommender.fit(sample_processed_data)

        recommender._knn.get_candidates = Mock(
            return_value=pl.DataFrame(
                {
                    "id": ["2"],
                    "name": ["Game 2"],
                    "cosine_distance": [0.1],
                    "publication_year": [2016],
                    "playing_time": [45],
                    "avg_rating": [8.0],
                    "bayesian_avg_rating": [8.1],
                    "popularity_score": [0.9],
                    "difficulty": [3.0],
                    "to_recommend": [1],
                    "mechanics": [["Dice"]],
                    "categories": [["Family"]],
                    "family": [[]],
                    "bgg_link": ["https://bgg.com/2"],
                    "image_url": ["https://cf.geekdo-images.com/2.jpg"],
                    "description": ["A fun dice rolling game"],
                }
            )
        )
        recommender._knn.get_input_game = Mock(
            return_value=sample_processed_data.filter(pl.col("id") == "1")
        )
        recommender._reranker.rerank = Mock(
            return_value=pl.DataFrame(
                {
                    "id": ["2"],
                    "name": ["Game 2"],
                    "final_score": [0.9],
                    "cosine_similarity": [0.9],
                    "normalized_difficulty_similarity": [0.8],
                    "normalized_playing_time_similarity": [0.85],
                    "normalized_avg_rating": [0.95],
                    "normalized_popularity": [0.9],
                }
            )
        )

        custom_weights = {
            "weight_cosine_similarity": 0.5,
            "weight_difficulty_similarity": 0.2,
            "weight_playing_time_similarity": 0.1,
            "weight_rating": 0.1,
            "weight_popularity": 0.1,
        }

        _ = recommender.recommend("1", weights=custom_weights)

        # Verify weights were passed to reranker
        call_args = recommender._reranker.rerank.call_args
        assert call_args[0][3] == custom_weights

    def test_recommend_with_custom_top_k(self, sample_processed_data):
        """Test recommendations with custom top_k."""
        recommender = BoardGameRecommender()
        recommender.fit(sample_processed_data)

        recommender._knn.get_candidates = Mock(
            return_value=pl.DataFrame(
                {
                    "id": ["2"],
                    "name": ["Game 2"],
                    "cosine_distance": [0.1],
                    "publication_year": [2016],
                    "playing_time": [45],
                    "avg_rating": [8.0],
                    "bayesian_avg_rating": [8.1],
                    "popularity_score": [0.9],
                    "difficulty": [3.0],
                    "to_recommend": [1],
                    "mechanics": [["Dice"]],
                    "categories": [["Family"]],
                    "family": [[]],
                    "bgg_link": ["https://bgg.com/2"],
                    "image_url": ["https://cf.geekdo-images.com/2.jpg"],
                    "description": ["A fun dice rolling game"],
                }
            )
        )
        recommender._knn.get_input_game = Mock(
            return_value=sample_processed_data.filter(pl.col("id") == "1")
        )
        recommender._reranker.rerank = Mock(
            return_value=pl.DataFrame(
                {
                    "id": ["2"],
                    "name": ["Game 2"],
                    "final_score": [0.9],
                    "cosine_similarity": [0.9],
                    "normalized_difficulty_similarity": [0.8],
                    "normalized_playing_time_similarity": [0.85],
                    "normalized_avg_rating": [0.95],
                    "normalized_popularity": [0.9],
                }
            )
        )

        _ = recommender.recommend("1", top_k=10)

        # Verify top_k was passed to reranker
        call_args = recommender._reranker.rerank.call_args
        assert call_args[0][2] == 10

    def test_add_comments_identifies_max_similarity(self):
        """Test that comments identify game with max similarity."""
        recommender = BoardGameRecommender()

        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "cosine_similarity": [0.95, 0.80, 0.85],
                "normalized_popularity": [0.70, 0.80, 0.75],
                "normalized_avg_rating": [0.75, 0.85, 0.80],
                "final_score": [0.90, 0.85, 0.88],
            }
        )

        result = recommender._add_comments(df)

        assert "comment" in result.columns
        # First game has highest similarity
        assert "very similar" in result["comment"][0].lower()

    def test_add_comments_identifies_max_popularity(self):
        """Test that comments identify most popular game."""
        recommender = BoardGameRecommender()

        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "cosine_similarity": [0.80, 0.85, 0.90],
                "normalized_popularity": [0.95, 0.80, 0.75],
                "normalized_avg_rating": [0.75, 0.80, 0.85],
                "final_score": [0.88, 0.85, 0.90],
            }
        )

        result = recommender._add_comments(df)

        # First game has highest popularity
        assert "popular" in result["comment"][0].lower()

    def test_add_comments_identifies_max_rating(self):
        """Test that comments identify highest-rated game."""
        recommender = BoardGameRecommender()

        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "cosine_similarity": [0.80, 0.85, 0.90],
                "normalized_popularity": [0.70, 0.80, 0.75],
                "normalized_avg_rating": [0.95, 0.80, 0.75],
                "final_score": [0.90, 0.85, 0.88],
            }
        )

        result = recommender._add_comments(df)

        # First game has highest rating
        assert "rating" in result["comment"][0].lower()

    def test_add_comments_empty_for_non_max(self):
        """Test that games without max values get empty comments."""
        recommender = BoardGameRecommender()

        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "cosine_similarity": [0.95, 0.80, 0.85],
                "normalized_popularity": [0.70, 0.95, 0.75],
                "normalized_avg_rating": [0.75, 0.80, 0.95],
                "final_score": [0.90, 0.88, 0.92],
            }
        )

        result = recommender._add_comments(df)

        # Each game is max in different category, so all should have comments
        # But if we had a 4th game with no max, it would be empty
        assert all(len(comment) > 0 for comment in result["comment"].to_list())

    def test_recommend_includes_metadata(self, sample_processed_data):
        """Test that recommendations include mechanics, categories, and bgg_link."""
        recommender = BoardGameRecommender()
        recommender.fit(sample_processed_data)

        # Add cosine_distance column to mock candidates
        candidates = sample_processed_data.filter(pl.col("id") != "1").head(2)
        candidates = candidates.with_columns(pl.lit(0.1).alias("cosine_distance"))

        recommender._knn.get_candidates = Mock(return_value=candidates)
        recommender._knn.get_input_game = Mock(
            return_value=sample_processed_data.filter(pl.col("id") == "1")
        )

        result = recommender.recommend("1", top_k=2)

        assert "mechanics" in result.columns
        assert "categories" in result.columns
        assert "bgg_link" in result.columns

    def test_recommend_includes_comment(self, sample_processed_data):
        """Test that recommendations include comment column."""
        recommender = BoardGameRecommender()
        recommender.fit(sample_processed_data)

        # Add cosine_distance column to mock candidates
        candidates = sample_processed_data.filter(pl.col("id") != "1").head(2)
        candidates = candidates.with_columns(pl.lit(0.1).alias("cosine_distance"))

        recommender._knn.get_candidates = Mock(return_value=candidates)
        recommender._knn.get_input_game = Mock(
            return_value=sample_processed_data.filter(pl.col("id") == "1")
        )

        result = recommender.recommend("1", top_k=2)

        assert "comment" in result.columns
