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
    try:
        movie_index = clean_dataset.loc[clean_dataset['title'].str.lower() == movie_name].index[0]
    except IndexError:
        return -1

    # Vectorization (columns are normalised by making lowercase
    overview_genres = clean_dataset['overview'].str.lower() + ' ' + clean_dataset['genres'].str.lower()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(overview_genres)

    # Compute cosine similarity with all others
    movie_vector = vectors[movie_index]
    similarity_scores = cosine_similarity(movie_vector, vectors).flatten()

    # Organises similarity scores to top n recs
    similarity_series = pd.Series(similarity_scores)
    similarity_series.sort_values(ascending=False, inplace=True)
    similarity_series.drop(movie_index, inplace=True)
    similar_indexes = similarity_series.index[:num_of_recs]

    # Represent recs in 2D array
    recommendation_array = []
    for index in range(0, len(similar_indexes)):
        recommendation_array.append([clean_dataset.loc[similar_indexes[index]]['title'], clean_dataset.loc[similar_indexes[index]]['overview']])

    return recommendation_array


# Outer program loop
another_movie = True
while another_movie:

    # Movie input
    movie_in = input("Please enter a movie title you would like recommendations based on:\n")
    movie_in = movie_in.lower().strip()
    # number of recs input
    valid_num_flag = False
    while not valid_num_flag:
        valid_num_flag = True
        try:
            num_recs_in = int(input("Please enter the number of recommendations you would like to receive:\n"))
        # Validation: integer, between 1 and 10
        except ValueError:
            print("Please enter a valid integer.")
            valid_num_flag = False
        else:
            if num_recs_in <=0:
                print("Please enter a positive integer.")
                valid_num_flag = False
            elif num_recs_in > 10:
                print("Maximum number of recommendations per movie is 10.")
                valid_num_flag = False

    # Runs calculation function
    recommendations = calculate_recommendations(movie_in, num_recs_in, '../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv')

    # Output
    print("----------------------------------------------------------------------------------------------------")
    # Movie not found
    if recommendations == -1:
        print("Movie not found in the dataset. Please try again.")
    # Expected result
    else:
        print("Your recommendations for movies similar to '" + movie_in + "' are:")
        for movie_num in range(0, num_recs_in):
            print("\nTitle: " + recommendations[movie_num][0] + "\nOverview: " + recommendations[movie_num][1])

    # User decides to loop again or not
    print("----------------------------------------------------------------------------------------------------")
    valid_another_flag = False
    while not valid_another_flag:
        valid_another_flag = True
        another_in = input("Enter Y to get recommendations for another movie or N to exit:\n").lower().strip()
        # Only accept Y/y/N/n inputs
        if another_in != 'y' and another_in != 'n':
            valid_another_flag = False
            print("Please enter a valid input of Y or N.")

    # End loop or format for new loop
    if another_in == 'n':
        another_movie = False
    else:
        print("----------------------------------------------------------------------------------------------------")
