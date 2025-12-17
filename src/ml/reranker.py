from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from src.config import settings


_RERANKER_DEFAULTS = settings.ML.get("reranker", {})


@dataclass
class ReRankerConfig:
    """Configuration for the ReRanker scoring weights.
    
    All weights should be non-negative. They will be used in a weighted linear
    combination to compute the final score for each candidate.
    
    Default values are loaded from config/settings.yaml under ml.reranker.
    
    Attributes:
        weight_cosine_similarity: Weight for KNN cosine similarity
        weight_year_similarity: Weight for publication year similarity
        weight_playing_time_similarity: Weight for playing time similarity
        weight_bayesian_rating: Weight for bayesian average rating
        weight_popularity: Weight for popularity score
        top_k: Number of recommendations to return
    """
    weight_cosine_similarity: float = _RERANKER_DEFAULTS.get("weight_cosine_similarity", 0.30)
    weight_year_similarity: float = _RERANKER_DEFAULTS.get("weight_year_similarity", 0.10)
    weight_playing_time_similarity: float = _RERANKER_DEFAULTS.get("weight_playing_time_similarity", 0.15)
    weight_bayesian_rating: float = _RERANKER_DEFAULTS.get("weight_bayesian_rating", 0.25)
    weight_popularity: float = _RERANKER_DEFAULTS.get("weight_popularity", 0.20)
    top_k: int = _RERANKER_DEFAULTS.get("top_k", 5)
    
    def __post_init__(self):
        """Validate that all weights are non-negative."""
        weights = [
            self.weight_cosine_similarity,
            self.weight_year_similarity,
            self.weight_playing_time_similarity,
            self.weight_bayesian_rating,
            self.weight_popularity,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
    
    @classmethod
    def from_dict(cls, config: dict) -> "ReRankerConfig":
        """Create config from a dictionary."""
        return cls(
            weight_cosine_similarity=config.get("weight_cosine_similarity", 0.30),
            weight_year_similarity=config.get("weight_year_similarity", 0.10),
            weight_playing_time_similarity=config.get("weight_playing_time_similarity", 0.15),
            weight_bayesian_rating=config.get("weight_bayesian_rating", 0.25),
            weight_popularity=config.get("weight_popularity", 0.20),
            top_k=config.get("top_k", 5),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "weight_cosine_similarity": self.weight_cosine_similarity,
            "weight_year_similarity": self.weight_year_similarity,
            "weight_playing_time_similarity": self.weight_playing_time_similarity,
            "weight_bayesian_rating": self.weight_bayesian_rating,
            "weight_popularity": self.weight_popularity,
            "top_k": self.top_k,
        }


# Column names used in the re-ranking process
class Columns:
    """Column name constants for the re-ranker."""
    # Input columns from KNN
    GAME_ID = "id"
    COSINE_DISTANCE = "cosine_distance"
    
    # Feature columns (already normalized in consolidation)
    PUBLICATION_YEAR = "publication_year"
    PLAYING_TIME = "playing_time"
    BAYESIAN_RATING = "bayesian_avg_rating"
    POPULARITY_SCORE = "popularity_score"
    
    # Output columns (normalized relative to input game)
    COSINE_SIMILARITY = "cosine_similarity"
    NORM_YEAR_SIMILARITY = "normalized_year_similarity"
    NORM_PLAYING_TIME_SIMILARITY = "normalized_playing_time_similarity"
    NORM_BAYESIAN_RATING = "normalized_bayesian_rating"
    NORM_POPULARITY = "normalized_popularity"
    FINAL_SCORE = "final_score"


class ReRanker:
    """Re-ranks KNN candidates using a weighted scoring system.
    
    The ReRanker takes candidates from KNN (with cosine distances) and computes
    a final score for each candidate based on multiple weighted features.
    All features are normalized to [0, 1] relative to the input game.
    
    Example:
        >>> config = ReRankerConfig(weight_cosine_similarity=0.4, top_k=5)
        >>> reranker = ReRanker(config)
        >>> recommendations = reranker.rerank(input_game_row, candidates_df)
    """
    
    def __init__(self):
        """Initialize the ReRanker with configuration.
        
        Args:
            config: ReRankerConfig instance. Uses defaults if None.
        """
        self.config = ReRankerConfig()
    
    def rerank(
        self, 
        input_game: pl.DataFrame, 
        candidates: pl.DataFrame
    ) -> pl.DataFrame:
        """Re-rank KNN candidates and return top-k recommendations.
        
        Args:
            input_game: Single-row DataFrame containing the input game's features.
            candidates: DataFrame of KNN candidates with cosine_distance column.
        
        Returns:
            DataFrame with top-k recommendations, sorted by final_score descending.
            Includes all normalized feature columns for radar visualization.
        """
        if input_game.height != 1:
            raise ValueError(f"input_game must have exactly 1 row, got {input_game.height}")
        
        if candidates.height == 0:
            return self._empty_result()
        
        # Extract input game feature values
        input_features = self._extract_features(input_game)
        
        # Compute normalized scores for each candidate
        scored = self._compute_scores(candidates, input_features)
        
        # Compute final weighted score
        scored = self._compute_final_score(scored)
        
        # Sort by final score and return top-k
        result = (
            scored
            .sort(Columns.FINAL_SCORE, descending=True)
            .head(self.config.top_k)
            .select(self._output_columns())
        )
        
        return result
    
    def _extract_features(self, game: pl.DataFrame) -> dict:
        """Extract feature values from a game row."""
        row = game.row(0, named=True)
        return {
            Columns.PUBLICATION_YEAR: row.get(Columns.PUBLICATION_YEAR),
            Columns.PLAYING_TIME: row.get(Columns.PLAYING_TIME),
            Columns.BAYESIAN_RATING: row.get(Columns.BAYESIAN_RATING),
            Columns.POPULARITY_SCORE: row.get(Columns.POPULARITY_SCORE),
        }
    
    def _compute_scores(self, candidates: pl.DataFrame, input_features: dict) -> pl.DataFrame:
        """Compute normalized scores for all candidates relative to input game."""
        
        # Convert cosine distance to similarity: similarity = 1 - distance
        # Cosine distance is in [0, 2], so similarity is in [-1, 1]
        # We normalize to [0, 1] by: (1 - distance + 1) / 2 = (2 - distance) / 2
        # Or simpler: since distance is typically [0, 1] for normalized vectors, 
        # similarity = 1 - distance is already [0, 1]
        
        input_year = input_features.get(Columns.PUBLICATION_YEAR)
        input_time = input_features.get(Columns.PLAYING_TIME)
        input_rating = input_features.get(Columns.BAYESIAN_RATING)
        input_popularity = input_features.get(Columns.POPULARITY_SCORE)
        
        # Get min/max from candidates for normalization
        stats = candidates.select([
            pl.col(Columns.PUBLICATION_YEAR).min().alias("year_min"),
            pl.col(Columns.PUBLICATION_YEAR).max().alias("year_max"),
            pl.col(Columns.PLAYING_TIME).min().alias("time_min"),
            pl.col(Columns.PLAYING_TIME).max().alias("time_max"),
            pl.col(Columns.BAYESIAN_RATING).min().alias("rating_min"),
            pl.col(Columns.BAYESIAN_RATING).max().alias("rating_max"),
            pl.col(Columns.POPULARITY_SCORE).min().alias("pop_min"),
            pl.col(Columns.POPULARITY_SCORE).max().alias("pop_max"),
            pl.col(Columns.COSINE_DISTANCE).min().alias("dist_min"),
            pl.col(Columns.COSINE_DISTANCE).max().alias("dist_max"),
        ]).row(0, named=True)
        
        return candidates.with_columns([
            # Cosine similarity: 1 - distance (already in [0, 1] for normalized vectors)
            (1.0 - pl.col(Columns.COSINE_DISTANCE)).alias(Columns.COSINE_SIMILARITY),
            
            # Year similarity: 1 - |candidate_year - input_year| / max_diff
            self._similarity_expr(
                Columns.PUBLICATION_YEAR, 
                input_year,
                stats["year_min"],
                stats["year_max"]
            ).alias(Columns.NORM_YEAR_SIMILARITY),
            
            # Playing time similarity: 1 - |candidate_time - input_time| / max_diff
            self._similarity_expr(
                Columns.PLAYING_TIME,
                input_time,
                stats["time_min"],
                stats["time_max"]
            ).alias(Columns.NORM_PLAYING_TIME_SIMILARITY),
            
            # Bayesian rating: normalize to [0, 1] within candidates
            self._normalize_expr(
                Columns.BAYESIAN_RATING,
                stats["rating_min"],
                stats["rating_max"]
            ).alias(Columns.NORM_BAYESIAN_RATING),
            
            # Popularity score: normalize to [0, 1] within candidates
            self._normalize_expr(
                Columns.POPULARITY_SCORE,
                stats["pop_min"],
                stats["pop_max"]
            ).alias(Columns.NORM_POPULARITY),
        ])
    
    def _similarity_expr(
        self, 
        col_name: str, 
        input_value: Optional[float],
        min_val: Optional[float],
        max_val: Optional[float]
    ) -> pl.Expr:
        """Create expression for similarity score (1 - normalized absolute difference)."""
        if input_value is None or min_val is None or max_val is None:
            return pl.lit(0.5)  # Default to neutral score if data missing
        
        max_diff = max_val - min_val
        if max_diff == 0:
            return pl.lit(1.0)  # All same value = perfect similarity
        
        # Similarity = 1 - |value - input| / max_possible_diff
        return (
            1.0 - (pl.col(col_name) - input_value).abs() / max_diff
        ).clip(0.0, 1.0)
    
    def _normalize_expr(
        self,
        col_name: str,
        min_val: Optional[float],
        max_val: Optional[float]
    ) -> pl.Expr:
        """Create expression for min-max normalization to [0, 1]."""
        if min_val is None or max_val is None:
            return pl.lit(0.5)
        
        range_val = max_val - min_val
        if range_val == 0:
            return pl.lit(0.5)
        
        return ((pl.col(col_name) - min_val) / range_val).clip(0.0, 1.0)
    
    def _compute_final_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute weighted final score for each candidate."""
        score_expr = (
            pl.col(Columns.COSINE_SIMILARITY) * self.config.weight_cosine_similarity +
            pl.col(Columns.NORM_YEAR_SIMILARITY) * self.config.weight_year_similarity +
            pl.col(Columns.NORM_PLAYING_TIME_SIMILARITY) * self.config.weight_playing_time_similarity +
            pl.col(Columns.NORM_BAYESIAN_RATING) * self.config.weight_bayesian_rating +
            pl.col(Columns.NORM_POPULARITY) * self.config.weight_popularity
        )
        
        return df.with_columns(score_expr.alias(Columns.FINAL_SCORE))
    
    def _output_columns(self) -> list[str]:
        """Return list of columns to include in output."""
        return [
            Columns.GAME_ID,
            "name",  # Include game name for display
            Columns.FINAL_SCORE,
            Columns.COSINE_SIMILARITY,
            Columns.NORM_YEAR_SIMILARITY,
            Columns.NORM_PLAYING_TIME_SIMILARITY,
            Columns.NORM_BAYESIAN_RATING,
            Columns.NORM_POPULARITY,
        ]
    
    def _empty_result(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame({
            Columns.GAME_ID: pl.Series([], dtype=pl.Utf8),
            "name": pl.Series([], dtype=pl.Utf8),
            Columns.FINAL_SCORE: pl.Series([], dtype=pl.Float64),
            Columns.COSINE_SIMILARITY: pl.Series([], dtype=pl.Float64),
            Columns.NORM_YEAR_SIMILARITY: pl.Series([], dtype=pl.Float64),
            Columns.NORM_PLAYING_TIME_SIMILARITY: pl.Series([], dtype=pl.Float64),
            Columns.NORM_BAYESIAN_RATING: pl.Series([], dtype=pl.Float64),
            Columns.NORM_POPULARITY: pl.Series([], dtype=pl.Float64),
        })
    
    def get_radar_data(self, recommendations: pl.DataFrame) -> list[dict]:
        """Convert recommendations to radar chart compatible format.
        
        Returns a list of dicts, each containing:
        - game_id: The game identifier
        - scores: Dict mapping feature names to normalized [0, 1] values
        
        This format is designed for future radar/hexagon visualization.
        """
        radar_features = [
            (Columns.COSINE_SIMILARITY, "Similarity"),
            (Columns.NORM_YEAR_SIMILARITY, "Year Match"),
            (Columns.NORM_PLAYING_TIME_SIMILARITY, "Time Match"),
            (Columns.NORM_BAYESIAN_RATING, "Rating"),
            (Columns.NORM_POPULARITY, "Popularity"),
        ]
        
        result = []
        for row in recommendations.iter_rows(named=True):
            scores = {
                display_name: row[col_name]
                for col_name, display_name in radar_features
            }
            result.append({
                "game_id": row[Columns.GAME_ID],
                "final_score": row[Columns.FINAL_SCORE],
                "scores": scores,
            })
        
        return result
