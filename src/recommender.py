import pandas as pd
import numpy as np
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
        
        # Calculamos la similitud entre todos los pares
        self.similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
        
        # Mapeo invertido: Título -> Índice de la fila
        self.indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        logger.info("Modelo 'fitted' exitosamente.")

    def get_recommendations(self, title: str, top_n: int = 5):
        """
        Devuelve las N películas más similares a un título dado con validación.
        """
        try:
            if self.indices is None or self.similarity_matrix is None:
                logger.error("El modelo no ha sido entrenado. Llama a .fit() primero.")
                return []

            if not title or not isinstance(title, str):
                logger.warning("Entrada de título inválida.")
                return []

            clean_query = title.strip()
            
            if clean_query not in self.indices:
                logger.warning(f"Película no encontrada: '{clean_query}'")
                return []

            # Obtener el índice (maneja duplicados si los hay)
            idx = self.indices[clean_query]
            movie_idx = idx if isinstance(idx, (int, np.int64)) else idx[0]

            # Calcular scores
            sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Tomar los top_n (excluyendo la película misma en el índice 0)
            actual_top = min(top_n, len(self.df) - 1)
            sim_scores = sim_scores[1:actual_top + 1]

            movie_indices = [i[0] for i in sim_scores]
            return self.df['title'].iloc[movie_indices].tolist()

        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}", exc_info=True)
            return []