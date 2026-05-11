from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Movie Recommender System"
    DATA_RAW_PATH: Path = Path("data/raw/movies.csv")
    DATA_PROCESSED_PATH: Path = Path("data/processed/movies_cleaned.csv")
    MODEL_PATH: Path = Path("models/recommender.pkl")
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()