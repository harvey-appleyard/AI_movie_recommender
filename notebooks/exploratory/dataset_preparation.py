import pandas as pd

df = pd.read_csv('../../data/raw/TMDB_IMDB_Movies_Dataset.csv')

cleaned = df[['id', 'title', 'genres', 'overview']]
cleaned.dropna(subset=['id', 'title', 'overview'])
cleaned.drop_duplicates(subset=['id'])

cleaned.to_csv('../../data/processed/Cleaned_Movies_Dataset.csv', index=False)

