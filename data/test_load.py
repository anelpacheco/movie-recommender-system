from src.data_loader import DataLoader
loader = DataLoader()
df = loader.load_movies()
print(df.head())