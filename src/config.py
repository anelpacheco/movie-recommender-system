from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Movie Recommender System"
    DATA_RAW_PATH: Path = Path("data/raw/movies.csv")
    DATA_PROCESSED_PATH: Path = Path("data/processed/movies_cleaned.csv")
    MODEL_PATH: Path = Path("models/recommender.pkl")
    LOG_LEVEL: str = "INFO"
    DEFAULT_SUMMARY_RATIO: float = 0.2  # Valor por defecto para evitar errores de argumentos

    # Nueva forma en Pydantic V2 de manejar el archivo .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()