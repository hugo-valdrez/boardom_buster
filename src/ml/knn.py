from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from src.config import settings


_KNN_DEFAULTS = settings.ML.get("knn", {})


@dataclass
class KNNConfig:
    """Configuration for the KNN candidate generator.
    
    Attributes:
        n_neighbors: Number of nearest neighbors to return (default: 50)
        metric: Distance metric to use (default: 'cosine')
        algorithm: Algorithm for nearest neighbor search (default: 'brute')
    """
    n_neighbors: int = _KNN_DEFAULTS.get("n_neighbors", 50)
    metric: str = _KNN_DEFAULTS.get("metric", "cosine")
    algorithm: str = _KNN_DEFAULTS.get("algorithm", "brute")
    
    def __post_init__(self):
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if self.metric not in ["cosine", "euclidean", "manhattan"]:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.algorithm not in ["brute", "ball_tree", "kd_tree", "auto"]:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


class KNNCandidateGenerator:
    """Generates candidate recommendations using K-Nearest Neighbors.
    
    Uses cosine distance (brute force) to find the most similar games
    based on their feature vectors (one-hot encoded categories, mechanics,
    player counts, etc.).
    
    Example:
        >>> knn = KNNCandidateGenerator()
        >>> knn.fit(processed_df, feature_columns)
        >>> candidates = knn.get_candidates(game_id="12345")
    """
    
    def __init__(self):
        """Initialize the KNN candidate generator.
        
        Args:
            config: KNNConfig instance. Uses defaults if None.
        """
        self.config = KNNConfig()
        self.excluded_columns = _KNN_DEFAULTS.get("excluded_columns", [
            "id", "thumbnail_url", "image_url", "description", "publication_year",
            "min_playing_time", "max_playing_time", "name", "num_ratings",
            "avg_rating", "stddev_rating", "to_recommend", "playing_time",
            "min_age", "bayesian_avg_rating", "popularity_score"
        ])
        self._model: Optional[NearestNeighbors] = None
        self._full_df: Optional[pl.DataFrame] = None  # All games (for input lookup)
        self._df: Optional[pl.DataFrame] = None  # Only recommendable games (for KNN)
        self._feature_columns: Optional[List[str]] = None
        self._id_to_idx: dict = {}  # Maps game_id to index in recommendable games
        self._idx_to_id: dict = {}  # Maps index to game_id in recommendable games
    
    def fit(
        self, 
        df: pl.DataFrame
    ) -> "KNNCandidateGenerator":
        """Fit the KNN model on the dataset.
        
        Args:
            df: DataFrame with game features (from consolidation).
        
        Returns:
            self for method chaining.
        """
        # Store full dataset (for looking up any game as input)
        self._full_df = df
        
        # Filter to only recommendable games for KNN model
        self._df = df.filter(pl.col("to_recommend") == 1)
        
        # Detect feature columns
        self._feature_columns = self._detect_feature_columns(self._df)
        
        # Build index mappings
        ids = df.select("id").to_series().to_list()
        self._id_to_idx = {game_id: idx for idx, game_id in enumerate(ids)}
        self._idx_to_id = {idx: game_id for idx, game_id in enumerate(ids)}
        
        # Extract feature matrix
        feature_matrix = df.select(self._feature_columns).to_numpy().astype(np.float32)
        
        # Fit the KNN model
        self._model = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors + 1, len(self._df)),  # +1 to exclude self
            metric=self.config.metric,
            algorithm=self.config.algorithm,
        )
        self._model.fit(feature_matrix)
        
        return self
    
    def _detect_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Auto-detect columns to use as features."""
        feature_cols = []
        
        for col in df.columns:
            # Skip excluded columns
            if col in self.excluded_columns:
                continue
            
            # Only include numeric columns
            dtype = df.schema[col]
            if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No feature columns detected. Check your data.")
        
        return feature_cols
    
    def get_candidates(
        self, 
        game_id: str,
        n_candidates: Optional[int] = None
    ) -> pl.DataFrame:
        """Get KNN candidates for a given game.
        
        The input game can be any game in the dataset (to_recommend=0 or 1),
        but only recommendable games (to_recommend=1) will be returned as candidates.
        
        Args:
            game_id: The ID of the input game.
            n_candidates: Number of candidates to return. Defaults to config.n_neighbors.
        
        Returns:
            DataFrame with candidate games and their cosine distances.
            Includes all original columns plus 'cosine_distance'.
        """
        # Get input game from full dataset (can be any game)
        input_game = self.get_input_game(game_id)
        
        n_candidates = n_candidates or self.config.n_neighbors
        
        # Get the feature vector for the input game
        input_features = (
            input_game
            .select(self._feature_columns)
            .row(0)
        )
        input_vector = np.array(input_features, dtype=np.float32).reshape(1, -1)
        
        # Find nearest neighbors (from recommendable games only)
        distances, indices = self._model.kneighbors(
            input_vector, 
            n_neighbors=min(n_candidates + 1, len(self._df))
        )
        
        # Flatten results
        distances = distances[0]
        indices = indices[0]
        
        # Get candidate game IDs and exclude the input game
        candidate_ids = [self._idx_to_id[i] for i in indices]
        
        # Filter out the input game from candidates (don't recommend the same game)
        filtered_ids = [cid for cid in candidate_ids if cid != game_id]
        filtered_distances = [dist for cid, dist in zip(candidate_ids, distances) if cid != game_id]
        
        # Take only the requested number after filtering
        candidate_ids = filtered_ids[:n_candidates]
        distances = filtered_distances[:n_candidates]
        
        # Build result DataFrame with all original columns
        candidates = self._df.filter(pl.col("id").is_in(candidate_ids))
        
        # Add cosine distance column
        # Create a mapping of id to distance
        id_to_dist = dict(zip(candidate_ids, distances))
        
        # Map distances to the dataframe - create a proper distance column
        distance_series = [id_to_dist[game_id] for game_id in candidates.select("id").to_series().to_list()]
        
        candidates = candidates.with_columns(
            pl.Series("cosine_distance", distance_series, dtype=pl.Float64)
        )
        
        return candidates
    
    def get_input_game(self, game_id: str) -> pl.DataFrame:
        """Get the full row for an input game.
        
        Can retrieve any game from the dataset, including non-recommendable games.
        
        Args:
            game_id: The ID of the game.
        
        Returns:
            Single-row DataFrame with all columns for the game.
        """        
        result = self._full_df.filter(pl.col("id") == game_id)
        
        if result.height == 0:
            raise ValueError(f"Game ID '{game_id}' not found in dataset.")
        
        return result
    
    @property
    def feature_columns(self) -> Optional[List[str]]:
        """Return the feature columns used for KNN."""
        return self._feature_columns
    
    @property 
    def n_games(self) -> int:
        """Return the number of games in the fitted model."""
        return len(self._id_to_idx) if self._id_to_idx else 0
