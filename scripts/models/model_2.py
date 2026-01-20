from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



def calculate_top_indexes(movie_name, num_of_recs):
    # Load dataset
    calc_dataset = pd.read_csv('../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv')

    # String safety after CSV load
    calc_dataset['overview'] = calc_dataset['overview'].fillna('').astype(str)
    calc_dataset['genres'] = calc_dataset['genres'].fillna('').astype(str)

    # Lookup
    try:
        movie_index = calc_dataset.loc[calc_dataset['title'].str.lower() == movie_name].index[0]
    except IndexError:
        return None

    # Vectorization (columns are normalised by making lowercase)
    overview_genres = calc_dataset['overview'].str.lower() + ' ' + calc_dataset['genres'].str.lower()
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

    return similar_indexes


# Outer program loop / UI
another_movie = True
while another_movie:

    # Movie input
    valid_movie_flag = False
    while not valid_movie_flag:
        valid_movie_flag = True
        movie_in = input("Please enter a movie title you would like recommendations based on:\n")
        movie_in = movie_in.lower().strip()
        # Validation: can't be empty string
        if movie_in == '' or movie_in is None:
            print("Please enter a valid movie title.")
            valid_movie_flag = False

    # Number of recs input
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
    recommendation_indexes = calculate_top_indexes(movie_in, num_recs_in)

    # Output
    print("----------------------------------------------------------------------------------------------------")
    # Movie not found
    if recommendation_indexes is None:
        print("Movie not found in the dataset. Please try again.")
    # Expected result
    else:
        # Load info dataset
        info_dataset = pd.read_csv('../../data/processed/Movie_Info_Dataset.csv')

        # Represent recs in 2D array
        recommendation_array = []
        for index in range(0, len(recommendation_indexes)):
            array_to_add = [info_dataset.loc[recommendation_indexes[index]]['title'],
                            info_dataset.loc[recommendation_indexes[index]]['tagline'],
                            info_dataset.loc[recommendation_indexes[index]]['release_date'],
                            info_dataset.loc[recommendation_indexes[index]]['runtime'],
                            info_dataset.loc[recommendation_indexes[index]]['overview'],
                            info_dataset.loc[recommendation_indexes[index]]['directors'],
                            info_dataset.loc[recommendation_indexes[index]]['cast']]

            # Validation: check for NaN, none, 0 and replace with unavailable
            for item in range(0, len(array_to_add)):
                if pd.isna(array_to_add[item]) or array_to_add[item] == "" or array_to_add[item] == 0:
                    array_to_add[item] = "Unavailable"
            recommendation_array.append(array_to_add)

        # Output movie recs
        print("Your recommendations for movies similar to '" + movie_in + "' are:")
        for movie_num in range(0, num_recs_in):
            print(f"\nTitle: {recommendation_array[movie_num][0]}"
                  f" ({recommendation_array[movie_num][2][:4]})"
                  f"\nTagline: {recommendation_array[movie_num][1]}"
                  f"\nRuntime: {recommendation_array[movie_num][3]} minutes"
                  f"\nOverview: {recommendation_array[movie_num][4]}"
                  f"\nDirectors: {recommendation_array[movie_num][5]}"
                  f"\nCast: {recommendation_array[movie_num][6]}")

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
