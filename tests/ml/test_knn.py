"""Tests for src/ml/knn.py"""
import pytest
import polars as pl

from src.ml.knn import KNNConfig, KNNCandidateGenerator


@pytest.fixture
def sample_game_data():
    """Fixture for sample game data."""
    return pl.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "name": ["Game 1", "Game 2", "Game 3", "Game 4", "Game 5"],
        "to_recommend": [1, 1, 1, 1, 0],  # Last game not recommendable
        "feature_1": [1.0, 0.8, 0.9, 0.7, 0.5],
        "feature_2": [0.5, 0.6, 0.5, 0.4, 0.3],
        "publication_year": [2015, 2016, 2017, 2018, 2019],
        "playing_time": [60, 45, 90, 30, 120],
        "avg_rating": [7.5, 8.0, 7.0, 6.5, 7.8],
        "popularity_score": [0.8, 0.9, 0.7, 0.6, 0.85],
        "mechanics": [["A"], ["B"], ["C"], ["D"], ["E"]],
        "categories": [["X"], ["Y"], ["Z"], ["W"], ["V"]],
        "bayesian_avg_rating": [7.6, 8.1, 7.1, 6.6, 7.9]
    })


class TestKNNConfig:
    """Test suite for the KNNConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KNNConfig()
        
        assert config.n_neighbors >= 1
        assert config.metric in ["cosine", "euclidean", "manhattan"]
        assert config.algorithm in ["brute", "ball_tree", "kd_tree", "auto"]
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = KNNConfig(n_neighbors=20, metric="euclidean", algorithm="ball_tree")
        
        assert config.n_neighbors == 20
        assert config.metric == "euclidean"
        assert config.algorithm == "ball_tree"
    
    def test_invalid_n_neighbors(self):
        """Test that n_neighbors < 1 raises error."""
        with pytest.raises(ValueError, match="n_neighbors must be at least 1"):
            KNNConfig(n_neighbors=0)
    
    def test_invalid_metric(self):
        """Test that unsupported metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            KNNConfig(metric="invalid")
    
    def test_invalid_algorithm(self):
        """Test that unsupported algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            KNNConfig(algorithm="invalid")


class TestKNNCandidateGenerator:
    """Test suite for the KNNCandidateGenerator class."""
    
    def test_initialization(self):
        """Test KNN candidate generator initialization."""
        knn = KNNCandidateGenerator()
        
        assert knn.config is not None
        assert knn._model is None
        assert knn._full_df is None
        assert knn._df is None
    
    def test_fit(self, sample_game_data):
        """Test fitting the KNN model."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        assert knn._full_df is not None
        assert knn._df is not None
        assert knn._model is not None
        assert knn._feature_columns is not None
        assert len(knn._id_to_idx) > 0
    
    def test_fit_filters_recommendable(self, sample_game_data):
        """Test that fit filters to only recommendable games."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        # Should only have 4 recommendable games (to_recommend=1)
        assert knn._df.height == 4
        assert all(knn._df["to_recommend"].to_list())
    
    def test_detect_feature_columns(self, sample_game_data):
        """Test automatic feature column detection."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        # Should include numeric features, exclude specified columns
        assert "feature_1" in knn._feature_columns
        assert "feature_2" in knn._feature_columns
        assert "id" not in knn._feature_columns
        assert "name" not in knn._feature_columns
        assert "mechanics" not in knn._feature_columns
    
    def test_get_candidates(self, sample_game_data):
        """Test getting KNN candidates for a game."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        candidates = knn.get_candidates("1", n_candidates=2)
        
        assert candidates.height <= 2
        assert "cosine_distance" in candidates.columns
        # Should not include the input game itself
        assert "1" not in candidates["id"].to_list()
    
    def test_get_candidates_excludes_input_game(self, sample_game_data):
        """Test that input game is excluded from candidates."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        candidates = knn.get_candidates("2", n_candidates=3)
        
        assert "2" not in candidates["id"].to_list()
    
    def test_get_candidates_only_recommendable(self, sample_game_data):
        """Test that only recommendable games are returned as candidates."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        candidates = knn.get_candidates("1", n_candidates=10)
        
        # Game 5 has to_recommend=0, should not be in candidates
        assert "5" not in candidates["id"].to_list()
        assert all(candidates["to_recommend"].to_list())
    
    def test_get_input_game(self, sample_game_data):
        """Test getting input game details."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        input_game = knn.get_input_game("1")
        
        assert input_game.height == 1
        assert input_game["id"][0] == "1"
    
    def test_get_candidates_invalid_game_id(self, sample_game_data):
        """Test error for invalid game ID."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        with pytest.raises((ValueError, KeyError)):
            knn.get_candidates("999")
    
    def test_get_candidates_before_fit(self):
        """Test error when getting candidates before fitting."""
        knn = KNNCandidateGenerator()
        
        with pytest.raises((RuntimeError, AttributeError)):
            knn.get_candidates("1")
    
    def test_get_candidates_with_custom_n(self, sample_game_data):
        """Test getting specific number of candidates."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        candidates = knn.get_candidates("1", n_candidates=1)
        
        assert candidates.height == 1
    
    def test_cosine_distances_in_range(self, sample_game_data):
        """Test that cosine distances are in valid range [0, 2]."""
        knn = KNNCandidateGenerator()
        knn.fit(sample_game_data)
        
        candidates = knn.get_candidates("1", n_candidates=2)
        
        distances = candidates["cosine_distance"].to_list()
        assert all(0 <= d <= 2 for d in distances)
    
    def test_method_chaining(self, sample_game_data):
        """Test that fit returns self for method chaining."""
        knn = KNNCandidateGenerator()
        result = knn.fit(sample_game_data)
        
        assert result is knn
