import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler

def setup_logger(name: str, log_file: str = "logs/app.log"):
    # Asegurar que la carpeta de logs existe
    Path("logs").mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Handler para consola (Rich)
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    
    # Handler para archivo (Rotativo: 5MB por archivo, máximo 3 archivos)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger("movie_recommender")