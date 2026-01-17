from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#Input
movie_name = input("Please enter a movie title you would like recommendations based on:\n")
movie_name = movie_name.lower().strip()

#Load
df = pd.read_csv('../../data/processed/Cleaned_Movies_Dataset.csv')

#String safety after CSV load
df['overview'] = df['overview'].fillna('').astype(str)
df['genres'] = df['genres'].fillna('').astype(str)

#Lookup
movie_index = df.loc[df['title'] == movie_name].index[0]

#Vectorization
overview_genres = df['overview'] + ' ' + df['genres']
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(overview_genres)

#Compute cosine similarity with all others
movie_vector = vectors[movie_index]
similarity_scores = cosine_similarity(movie_vector, vectors).flatten()

similarity_series = pd.Series(similarity_scores)
similarity_series = similarity_series.sort_values(ascending=False)
similarity_series = similarity_series.drop(movie_index)

similar_indexes = similarity_series.index[:5]

for index in range(0, len(similar_indexes)):
    print("Title: " + df.loc[similar_indexes[index]]['title'] + ", Overview: " + df.loc[similar_indexes[index]]['overview'])