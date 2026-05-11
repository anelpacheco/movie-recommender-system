import pytest
import pandas as pd
import numpy as np
from src.recommender import MovieRecommender
from src.preprocessing import TextPreprocessor

@pytest.fixture
def sample_data():
    """Crea un pequeño DataFrame para pruebas."""
    data = {
        'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale'],
        'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance']
    }
    return pd.DataFrame(data)

@pytest.fixture
def trained_recommender(sample_data):
    """Devuelve un recomendador ya entrenado con datos de muestra."""
    processor = TextPreprocessor()
    recommender = MovieRecommender()
    
    processed_df = processor.create_soup(sample_data)
    matrix = processor.get_feature_matrix(processed_df)
    recommender.fit(processed_df, matrix)
    
    return recommender

def test_recommendations_length(trained_recommender):
    """Verifica que el número de recomendaciones sea el correcto."""
    results = trained_recommender.get_recommendations('Toy Story', top_n=2)
    assert len(results) == 2

def test_invalid_movie_title(trained_recommender):
    """Verifica que retorne lista vacía si la película no existe."""
    results = trained_recommender.get_recommendations('The Matrix')
    assert results == []

def test_empty_input(trained_recommender):
    """Verifica que maneje entradas vacías."""
    assert trained_recommender.get_recommendations("") == []
    assert trained_recommender.get_recommendations(None) == []