from typing import Optional, List

import polars as pl

from src.ml.knn import KNNCandidateGenerator, KNNConfig
from src.ml.reranker import ReRanker, ReRankerConfig
from src.other.abstract.write_reader_factory import WriterReaderFactory
from src.config import settings


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
    ) -> pl.DataFrame:
        """Get recommendations for a game.
        
        Args:
            game_id: The ID of the input game.
            n_candidates: Number of KNN candidates to consider. Defaults to config.
            top_k: Number of final recommendations. Defaults to reranker config.
        
        Returns:
            DataFrame with top-k recommendations, including:
            - id, final_score, cosine_similarity
            - normalized_year_similarity, normalized_playing_time_similarity
            - normalized_bayesian_rating, normalized_popularity
        """
        if not self._is_fitted:
            raise RuntimeError("Recommender not fitted. Call fit() or load_data() first.")
        
        # Get KNN candidates
        candidates = self._knn.get_candidates(game_id, n_candidates)
        
        # Get input game for re-ranking context
        input_game = self._knn.get_input_game(game_id)
        
        # Re-rank and return top-k
        if top_k is not None:
            result = self._reranker.rerank(input_game, candidates, top_k)
        else:
            result = self._reranker.rerank(input_game, candidates, self._reranker.config.top_k)
        
        return result
    
    def get_game_info(self, game_id: str) -> dict:
        """Get basic info for a game.
        
        Args:
            game_id: The game ID.
        
        Returns:
            Dict with game information.
        """
        game = self._knn.get_input_game(game_id)
        return game.row(0, named=True)
    
    # debuging purposes
    @property
    def n_games(self) -> int:
        """Number of games in the model."""
        return self._knn.n_games
    
    # debuging purposes
    @property
    def feature_columns(self) -> Optional[List[str]]:
        """Feature columns used for KNN."""
        return self._knn.feature_columns
