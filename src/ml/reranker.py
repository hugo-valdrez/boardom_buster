from dataclasses import dataclass
from typing import Optional, Union

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
        weight_difficulty_similarity: Weight for difficulty similarity
        weight_playing_time_similarity: Weight for playing time similarity
        weight_rating: Weight for rating (uses bayesian if available, otherwise average)
        weight_popularity: Weight for popularity score
        top_k: Number of recommendations to return
    """

    weight_cosine_similarity: float = _RERANKER_DEFAULTS.get("weight_cosine_similarity", 0.40)
    weight_difficulty_similarity: float = _RERANKER_DEFAULTS.get(
        "weight_difficulty_similarity", 0.20
    )
    weight_playing_time_similarity: float = _RERANKER_DEFAULTS.get(
        "weight_playing_time_similarity", 0.10
    )
    weight_rating: float = _RERANKER_DEFAULTS.get("weight_rating", 0.15)
    weight_popularity: float = _RERANKER_DEFAULTS.get("weight_popularity", 0.15)
    top_k: int = _RERANKER_DEFAULTS.get("top_k", 5)

    def __post_init__(self):
        """Validate that all weights are non-negative."""
        weights = [
            self.weight_cosine_similarity,
            self.weight_difficulty_similarity,
            self.weight_playing_time_similarity,
            self.weight_rating,
            self.weight_popularity,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "weight_cosine_similarity": self.weight_cosine_similarity,
            "weight_difficulty_similarity": self.weight_difficulty_similarity,
            "weight_playing_time_similarity": self.weight_playing_time_similarity,
            "weight_rating": self.weight_rating,
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
    DIFFICULTY = "difficulty"
    PLAYING_TIME = "playing_time"
    BAYESIAN_RATING = "bayesian_avg_rating"
    AVG_RATING = "avg_rating"
    POPULARITY_SCORE = "popularity_score"

    # Output columns (normalized relative to input game)
    COSINE_SIMILARITY = "cosine_similarity"
    NORM_DIFFICULTY_SIMILARITY = "normalized_difficulty_similarity"
    NORM_PLAYING_TIME_SIMILARITY = "normalized_playing_time_similarity"
    NORM_AVG_RATING = "normalized_avg_rating"
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
        self, input_game: pl.DataFrame, candidates: pl.DataFrame, top_k: int, weights: dict
    ) -> pl.DataFrame:
        """Re-rank KNN candidates and return top-k recommendations.

        Args:
            input_game: Single-row DataFrame containing the input game's features.
            candidates: DataFrame of KNN candidates with cosine_distance column.

        Returns:
            DataFrame with top-k recommendations, sorted by final_score descending.
        """
        if input_game.height != 1:
            raise ValueError(f"input_game must have exactly 1 row, got {input_game.height}")

        # Extract input game feature values
        input_features = self._extract_features(input_game)

        # Compute normalized scores for each candidate
        scored = self._compute_scores(candidates, input_features)

        # Compute final weighted score
        scored = self._compute_final_score(scored, weights)

        # Sort by final score descending
        scored = scored.sort(Columns.FINAL_SCORE, descending=True)

        # Deduplicate by family: keep only the best-scoring game from each family
        scored = self._deduplicate_by_family(scored)

        # Return top-k
        result = scored.head(top_k).select(self._output_columns())
        return result

    def _extract_features(self, game: pl.DataFrame) -> dict:
        """Extract feature values from a game row."""
        row = game.row(0, named=True)

        # Use bayesian_avg_rating if available (and non-zero), fall back to avg_rating
        rating = row.get(Columns.BAYESIAN_RATING)
        if rating == 0:
            rating = row.get(Columns.AVG_RATING)

        return {
            Columns.DIFFICULTY: row.get(Columns.DIFFICULTY),
            Columns.PLAYING_TIME: row.get(Columns.PLAYING_TIME),
            Columns.BAYESIAN_RATING: rating,
            Columns.POPULARITY_SCORE: row.get(Columns.POPULARITY_SCORE),
        }

    def _compute_scores(self, candidates: pl.DataFrame, input_features: dict) -> pl.DataFrame:
        """Compute normalized scores for all candidates relative to input game."""

        input_difficulty = input_features.get(Columns.DIFFICULTY)
        input_time = input_features.get(Columns.PLAYING_TIME)
        input_rating_expr = (
            pl.when(
                pl.col(Columns.BAYESIAN_RATING).is_not_null()
                & (pl.col(Columns.BAYESIAN_RATING) != 0)
            )
            .then(pl.col(Columns.BAYESIAN_RATING))
            .otherwise(pl.col(Columns.AVG_RATING))
        )

        # Get min/max from candidates for normalization
        stats = candidates.select(
            [
                pl.col(Columns.DIFFICULTY).min().alias("difficulty_min"),
                pl.col(Columns.DIFFICULTY).max().alias("difficulty_max"),
                pl.col(Columns.PLAYING_TIME).min().alias("time_min"),
                pl.col(Columns.PLAYING_TIME).max().alias("time_max"),
                input_rating_expr.min().alias("rating_min"),
                input_rating_expr.max().alias("rating_max"),
                pl.col(Columns.POPULARITY_SCORE).min().alias("pop_min"),
                pl.col(Columns.POPULARITY_SCORE).max().alias("pop_max"),
                pl.col(Columns.COSINE_DISTANCE).min().alias("dist_min"),
                pl.col(Columns.COSINE_DISTANCE).max().alias("dist_max"),
            ]
        ).row(0, named=True)

        return candidates.with_columns(
            [
                # Convert cosine distance to similarity: similarity = 1 - distance
                (1.0 - pl.col(Columns.COSINE_DISTANCE)).alias(Columns.COSINE_SIMILARITY),
                # Difficulty similarity: 1 - |candidate_difficulty - input_difficulty| / max_diff
                self._similarity_expr(
                    Columns.DIFFICULTY,
                    input_difficulty,
                    stats["difficulty_min"],
                    stats["difficulty_max"],
                ).alias(Columns.NORM_DIFFICULTY_SIMILARITY),
                # Playing time similarity: 1 - |candidate_time - input_time| / max_diff
                self._similarity_expr(
                    Columns.PLAYING_TIME, input_time, stats["time_min"], stats["time_max"]
                ).alias(Columns.NORM_PLAYING_TIME_SIMILARITY),
                # Rating: normalize to [0, 1] within candidates
                self._normalize_expr(
                    input_rating_expr, stats["rating_min"], stats["rating_max"]
                ).alias(Columns.NORM_AVG_RATING),
                # Popularity score: normalize to [0, 1] within candidates
                self._normalize_expr(
                    Columns.POPULARITY_SCORE, stats["pop_min"], stats["pop_max"]
                ).alias(Columns.NORM_POPULARITY),
            ]
        )

    def _similarity_expr(
        self,
        col_name: str,
        input_value: Optional[float],
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> pl.Expr:
        """Create expression for similarity score (1 - normalized absolute difference)."""
        max_diff = max_val - min_val
        if max_diff == 0:
            return pl.lit(1.0)  # All same value = perfect similarity

        # Similarity = 1 - |value - input| / max_possible_diff
        return (1.0 - (pl.col(col_name) - input_value).abs() / max_diff).clip(0.0, 1.0)

    def _normalize_expr(
        self,
        expr_or_name: Union[str, pl.Expr],  # Updated type hint
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> pl.Expr:
        """Create expression for min-max normalization to [0, 1]."""
        range_val = max_val - min_val
        if range_val == 0:
            return pl.lit(0.5)

        col_expr = pl.col(expr_or_name) if isinstance(expr_or_name, str) else expr_or_name

        return ((col_expr - min_val) / range_val).clip(0.0, 1.0)

    def _compute_final_score(self, df: pl.DataFrame, weights: dict) -> pl.DataFrame:
        """Compute weighted final score for each candidate.

        Args:
            df: DataFrame with normalized score columns.
            weights: Dict with weight values.
        """
        w_cosine = weights["weight_cosine_similarity"]
        w_difficulty = weights["weight_difficulty_similarity"]
        w_time = weights["weight_playing_time_similarity"]
        w_rating = weights["weight_rating"]
        w_popularity = weights["weight_popularity"]

        score_expr = (
            pl.col(Columns.COSINE_SIMILARITY) * w_cosine
            + pl.col(Columns.NORM_DIFFICULTY_SIMILARITY) * w_difficulty
            + pl.col(Columns.NORM_PLAYING_TIME_SIMILARITY) * w_time
            + pl.col(Columns.NORM_AVG_RATING) * w_rating
            + pl.col(Columns.NORM_POPULARITY) * w_popularity
        )

        return df.with_columns(score_expr.alias(Columns.FINAL_SCORE))

    def _deduplicate_by_family(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep only the highest-scoring game from each family.

        Games without a family (null or empty) are all kept.
        For games with families, only the first occurrence (highest score) is kept
        for each unique family.

        Args:
            df: DataFrame sorted by final_score descending.

        Returns:
            DataFrame with at most one game per family.
        """
        seen_families: set = set()
        rows_to_keep: list[int] = []

        for idx, row in enumerate(df.iter_rows(named=True)):
            family_value = row.get("family")

            # If no family, always keep
            if family_value is None or len(family_value) == 0:
                rows_to_keep.append(idx)
                continue

            # Check if any of this game's families have been seen
            game_families = set(family_value)
            if game_families.isdisjoint(seen_families):
                # No overlap with seen families, keep this game
                rows_to_keep.append(idx)
                seen_families.update(game_families)

        return df[rows_to_keep]

    def _output_columns(self) -> list[str]:
        """Return list of columns to include in output."""
        return [
            Columns.GAME_ID,
            "name",  # Include game name for display
            Columns.FINAL_SCORE,
            Columns.COSINE_SIMILARITY,
            Columns.NORM_DIFFICULTY_SIMILARITY,
            Columns.NORM_PLAYING_TIME_SIMILARITY,
            Columns.NORM_AVG_RATING,
            Columns.NORM_POPULARITY,
        ]
