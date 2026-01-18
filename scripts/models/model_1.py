from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#Input
movie_name = input("Please enter a movie title you would like recommendations based on:\n")
movie_name = movie_name.lower().strip()

#Load
clean_dataset = pd.read_csv('../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv')

#String safety after CSV load
clean_dataset['overview'] = clean_dataset['overview'].fillna('').astype(str)
clean_dataset['genres'] = clean_dataset['genres'].fillna('').astype(str)

#Lookup
movie_index = clean_dataset.loc[clean_dataset['title'].str.lower() == movie_name].index[0]

#Vectorization (columns are normalised by making lowercase
overview_genres = clean_dataset['overview'].str.lower() + ' ' + clean_dataset['genres'].str.lower()
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(overview_genres)

#Compute cosine similarity with all others
movie_vector = vectors[movie_index]
similarity_scores = cosine_similarity(movie_vector, vectors).flatten()

similarity_series = pd.Series(similarity_scores)
similarity_series = similarity_series.sort_values(ascending=False)
similarity_series = similarity_series.drop(movie_index)

similar_indexes = similarity_series.index[:5]

print("----------------------------------------------------------------------------------------------------")
print("Your recommendations for films similar to '" + movie_name + "' are:")
for index in range(0, len(similar_indexes)):
    print("\nTitle: " + clean_dataset.loc[similar_indexes[index]]['title'] + "\nOverview: " + clean_dataset.loc[similar_indexes[index]]['overview'])
