from src.logger import logger

class MovieRecommender:
    def __init__(self):
        self.model = None
        logger.info("MovieRecommender initialized")

    def train(self, data):
        """Entrena el modelo de recomendación."""
        pass

    def get_recommendations(self, movie_title: str, top_n: int = 5):
        """Devuelve una lista de recomendaciones."""
        pass