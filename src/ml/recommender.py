from typing import Optional

import polars as pl

from src.config import settings
from src.ml.knn import KNNCandidateGenerator
from src.ml.reranker import ReRanker
from src.other.abstract.write_reader_factory import WriterReaderFactory


class BoardGameRecommender:
    """High-level recommender combining KNN candidate generation and re-ranking.

    Example:
        >>> recommender = BoardGameRecommender()
        >>> recommender.load_data()  # or recommender.fit(df)
        >>> recommendations = recommender.recommend("12345")
    """

    def __init__(self):
        """Initialize the recommender.

        Args:
            knn_config: Configuration for KNN. Uses defaults from settings.yaml if None.
            reranker_config: Configuration for ReRanker. Uses defaults from settings.yaml if None.
        """
        self._knn = KNNCandidateGenerator()
        self._reranker = ReRanker()
        self._is_fitted = False

    def load_data(self) -> "BoardGameRecommender":
        """Load processed data and fit the model.

        Returns:
            self for method chaining.
        """
        data_dir = settings.PATHS["processed_data"]
        writer_reader = WriterReaderFactory.create_from_directory(data_dir)
        df = writer_reader.read()

        return self.fit(df)

    def fit(self, df: pl.DataFrame) -> "BoardGameRecommender":
        """Fit the recommender on a DataFrame.

        Args:
            df: Processed DataFrame from consolidation.

        Returns:
            self for method chaining.
        """
        self._knn.fit(df)
        self._is_fitted = True
        return self

    def recommend(
        self,
        game_id: str,
        n_candidates: Optional[int] = None,
        top_k: Optional[int] = None,
        weights: Optional[dict] = None,
        exclude_same_family: bool = True,
    ) -> pl.DataFrame:
        """Get recommendations for a game.

        Args:
            game_id: The ID of the input game.
            n_candidates: Number of KNN candidates to consider. Defaults to config.
            top_k: Number of final recommendations. Defaults to reranker config.
            weights: Custom weights for reranking. Defaults to reranker config.
            exclude_same_family: If True, exclude games that share any family with the input game.
                Defaults to True.

        Returns:
            DataFrame with top-k recommendations, including:
            - id, final_score, cosine_similarity
            - normalized_difficulty_similarity, normalized_playing_time_similarity
            - normalized_bayesian_rating, normalized_popularity
        """
        if not self._is_fitted:
            raise RuntimeError("Recommender not fitted. Call fit() or load_data() first.")

        # Get KNN candidates
        candidates = self._knn.get_candidates(game_id, n_candidates, exclude_same_family)

        # Get input game for re-ranking context
        input_game = self._knn.get_input_game(game_id)

        # Re-rank and return top-k
        top_k = top_k or self._reranker.config.top_k
        weights = weights or self._reranker.config.to_dict()
        result = self._reranker.rerank(input_game, candidates, top_k, weights)

        # Join additional columns from candidates
        additional_cols = candidates.select(
            ["id", "mechanics", "categories", "bgg_link", "image_url", "description"]
        )
        result = result.join(additional_cols, on="id", how="left")

        # Generate comments for top performers in each category
        result = self._add_comments(result)

        return result

    def _add_comments(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add comments highlighting what each game excels at.

        Args:
            df: DataFrame with recommendation scores.

        Returns:
            DataFrame with added 'comment' column.
        """
        # Find the game with max for each metric (breaking ties by final_score)
        max_sim_game = df.sort(["cosine_similarity", "final_score"], descending=True).row(
            0, named=True
        )
        max_pop_game = df.sort(["normalized_popularity", "final_score"], descending=True).row(
            0, named=True
        )
        max_rating_game = df.sort(["normalized_avg_rating", "final_score"], descending=True).row(
            0, named=True
        )

        comments = []

        for row in df.iter_rows(named=True):
            # Determine which facts are true for this specific game
            is_sim = row["id"] == max_sim_game["id"]
            is_pop = row["id"] == max_pop_game["id"]
            is_rating = row["id"] == max_rating_game["id"]

            parts = []

            # Handle Similarity (Always stands alone)
            if is_sim:
                parts.append("is very similar to your game")

            # We construct a specific phrase depending on the combination
            if is_pop and is_rating:
                parts.append("is the most popular and highest-rated recommendation")
            elif is_pop:
                parts.append("is the most popular recommendation")
            elif is_rating:
                parts.append("has the best rating across all recommendations")

            # Construct the sentence
            if not parts:
                comments.append("")
            else:
                full_sentence = "This game " + " and ".join(parts) + "!"
                comments.append(full_sentence)

        return df.with_columns(pl.Series("comment", comments, dtype=pl.Utf8))
