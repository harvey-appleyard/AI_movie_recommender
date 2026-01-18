from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



def calculate_recommendations(movie_name, num_of_recs, dataset_file_path):
    # Load dataset
    clean_dataset = pd.read_csv(dataset_file_path)

    # String safety after CSV load
    clean_dataset['overview'] = clean_dataset['overview'].fillna('').astype(str)
    clean_dataset['genres'] = clean_dataset['genres'].fillna('').astype(str)

    # Lookup
    movie_index = clean_dataset.loc[clean_dataset['title'].str.lower() == movie_name].index[0]

    # Vectorization (columns are normalised by making lowercase
    overview_genres = clean_dataset['overview'].str.lower() + ' ' + clean_dataset['genres'].str.lower()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(overview_genres)

    # Compute cosine similarity with all others
    movie_vector = vectors[movie_index]
    similarity_scores = cosine_similarity(movie_vector, vectors).flatten()

    similarity_series = pd.Series(similarity_scores)
    similarity_series.sort_values(ascending=False, inplace=True)
    similarity_series.drop(movie_index, inplace=True)

    similar_indexes = similarity_series.index[:num_of_recs]
    recommendation_array = []
    for index in range(0, len(similar_indexes)):
        recommendation_array.append([clean_dataset.loc[similar_indexes[index]]['title'], clean_dataset.loc[similar_indexes[index]]['overview']])

    return recommendation_array



# Input
movie_in = input("Please enter a movie title you would like recommendations based on:\n")
movie_in = movie_in.lower().strip()
num_recs_in = int(input("Please enter the number of recommendations you would like to receive:\n"))

#Runs calculation function
recommendations = calculate_recommendations(movie_in, num_recs_in, '../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv')

# Output
print("----------------------------------------------------------------------------------------------------")
print("Your recommendations for movies similar to '" + movie_in + "' are:")
for movie_num in range(0, num_recs_in):
    print("\nTitle: " + recommendations[movie_num][0] + "\nOverview: " + recommendations[movie_num][1])
