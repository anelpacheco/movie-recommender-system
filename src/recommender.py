import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import logger

class MovieRecommender:
    def __init__(self):
        self.similarity_matrix = None
        self.indices = None
        self.df = None

    def fit(self, df: pd.DataFrame, feature_matrix):
        """
        Calcula la matriz de similitud y crea el mapeo de índices.
        """
        logger.info("Calculando matriz de similitud coseno...")
        self.df = df
        
        # Calculamos la similitud entre todos los pares (Matriz N x N)
        self.similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
        
        # Mapeo invertido: Título -> Índice de la fila
        # Eliminamos duplicados para evitar colisiones en la búsqueda
        self.indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        logger.info("Modelo entrenado exitosamente.")

    def get_recommendations(self, title: str, top_n: int = 5):
        """
        Devuelve las N películas más similares a un título dado.
        """
        if self.indices is None or title not in self.indices:
            logger.warning(f"La película '{title}' no se encuentra en el dataset.")
            return []

        # Obtener el índice de la película
        idx = self.indices[title]

        # Obtener puntuaciones de similitud de esa película con todas las demás
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Ordenar de mayor a menor similitud (el índice 0 es la película misma)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Seleccionar los mejores N (excluyendo el primero)
        sim_scores = sim_scores[1:top_n + 1]

        # Obtener los índices de las películas
        movie_indices = [i[0] for i in sim_scores]

        # Retornar los títulos
        return self.df['title'].iloc[movie_indices].tolist()