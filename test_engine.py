from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.recommender import MovieRecommender

# Pipeline
loader = DataLoader()
processor = TextPreprocessor()
recommender = MovieRecommender()

movies = loader.load_movies()
processed_df = processor.create_soup(movies)
matrix = processor.get_feature_matrix(processed_df)

# Entrenamiento
recommender.fit(processed_df, matrix)

# Test
results = recommender.get_recommendations("Toy Story (1995)")
print(f"Recomendaciones para Toy Story: {results}")