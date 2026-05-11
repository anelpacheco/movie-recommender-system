import pandas as pd
from pathlib import Path
from src.logger import logger
from src.config import settings

class DataLoader:
    def __init__(self):
        self.raw_path = Path("data/raw")
        
    def load_movies(self) -> pd.DataFrame:
        """Carga el dataset de películas y valida su estructura."""
        file_path = self.raw_path / "movies.csv"
        
        if not file_path.exists():
            logger.error(f"Dataset no encontrado en {file_path}")
            raise FileNotFoundError(f"Falta archivo crítico: {file_path}")
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset cargado exitosamente: {len(df)} películas encontradas.")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise

    def load_tags(self) -> pd.DataFrame:
        """Carga los tags para enriquecer el contenido."""
        file_path = self.raw_path / "tags.csv"
        return pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()