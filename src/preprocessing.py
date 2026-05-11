import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logger

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')

    def clean_text(self, text: str) -> str:
        """Limpia el texto: minúsculas, quita caracteres especiales y espacios extra."""
        if not isinstance(text, str):
            return ""
        # Convertir a minúsculas
        text = text.lower()
        # Quitar el año del título (ej: "Toy Story (1995)" -> "toy story")
        text = re.sub(r'\(\d{4}\)', '', text)
        # Quitar caracteres especiales y limpiar espacios
        text = re.sub(r'[^a-z0-9\s|]', '', text)
        return text.strip()

    def create_soup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea una columna 'soup' que combina géneros y título procesado."""
        logger.info("Creando 'metadata soup' para vectorización...")
        
        # Procesar géneros: 'Action|Adventure' -> 'action adventure'
        df['genres_cleaned'] = df['genres'].str.replace('|', ' ', regex=False).str.lower()
        df['title_cleaned'] = df['title'].apply(self.clean_text)
        
        # Combinar todo en una sopa de metadatos
        df['metadata_soup'] = df['title_cleaned'] + " " + df['genres_cleaned']
        
        return df

    def get_feature_matrix(self, df: pd.DataFrame):
        """Genera la matriz de conteo a partir de la columna 'metadata_soup'."""
        logger.info("Generando matriz de características (CountVectorizer)...")
        count_matrix = self.vectorizer.fit_transform(df['metadata_soup'])
        return count_matrix