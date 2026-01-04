"""Tests for src/app/dependencies.py"""

from unittest.mock import Mock, patch

import pytest

from src.app.dependencies import RecommenderState, get_recommender, state
from src.ml.recommender import BoardGameRecommender


class TestRecommenderState:
    """Test suite for the RecommenderState class."""

    def test_initialization(self):
        """Test that RecommenderState initializes with None recommender."""
        state = RecommenderState()
        assert state.recommender is None

    @patch("src.app.dependencies.BoardGameRecommender")
    def test_initialize(self, mock_recommender_class):
        """Test the initialize method creates and loads recommender."""
        mock_recommender = Mock()
        mock_recommender._knn._df = [1, 2, 3]  # Mock list for len()
        mock_recommender._knn.df_size = 3
        mock_recommender_class.return_value = mock_recommender

        state = RecommenderState()
        state.initialize()

        assert state.recommender == mock_recommender
        mock_recommender.load_data.assert_called_once()

    def test_reload(self):
        """Test the reload method reloads the recommender data."""
        state = RecommenderState()
        state.recommender = Mock()

        state.reload()

        state.recommender.load_data.assert_called_once()

    def test_reload_without_initialization(self):
        """Test reload raises error when recommender is not initialized."""
        state = RecommenderState()

        with pytest.raises(AttributeError):
            state.reload()


class TestGetRecommender:
    """Test suite for the get_recommender dependency function."""

    def test_get_recommender_returns_instance(self):
        """Test that get_recommender returns the state's recommender."""
        mock_recommender = Mock(spec=BoardGameRecommender)
        original_recommender = state.recommender

        try:
            state.recommender = mock_recommender
            result = get_recommender()
            assert result == mock_recommender
        finally:
            state.recommender = original_recommender
