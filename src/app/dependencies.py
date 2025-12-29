from src.ml.recommender import BoardGameRecommender


class RecommenderState:
    """Holds the global recommender instance."""

    def __init__(self):
        self.recommender: BoardGameRecommender = None

    def initialize(self):
        """Initialize and load the recommender."""
        self.recommender = BoardGameRecommender()
        self.recommender.load_data()

    def reload(self):
        """Reload the recommender with fresh data."""
        self.recommender.load_data()


state = RecommenderState()


def get_recommender() -> BoardGameRecommender:
    """Dependency to get the recommender instance."""
    return state.recommender
