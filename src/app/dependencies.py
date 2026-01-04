from src.ml.recommender import BoardGameRecommender


class RecommenderState:
    """Holds the global recommender instance."""

    def __init__(self):
        self.recommender: BoardGameRecommender = None
        self._initialized = False

    def initialize(self):
        """Initialize and load the recommender."""
        if self._initialized:
            return

        print("Initializing BoardGameRecommender...")
        self.recommender = BoardGameRecommender()
        print("Loading game data...")
        self.recommender.load_data()
        print(f"Recommender ready! Loaded {len(self.recommender._knn._df)} games")
        print(f"- Recommendable games: {self.recommender._knn.df_size}")
        self._initialized = True

    def reload(self):
        """Reload the recommender with fresh data."""
        self.recommender.load_data()


state = RecommenderState()


def get_recommender() -> BoardGameRecommender:
    """Dependency to get the recommender instance. Initializes on first call (lazy loading)."""
    if not state._initialized:
        state.initialize()
    return state.recommender
