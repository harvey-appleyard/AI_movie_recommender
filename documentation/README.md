# AI_movie_recommender

A content-based movie recommendation system that uses Natural Language Processing (NLP) techniques to suggest movies
similar to a given input title. The system analyzes movie keywords, genres, and overviews using TF-IDF vectorization and
cosine similarity, and enhances recommendations with confidence-weighted ratings.

---

Features:
- Content-based recommendations using text similarity
- TF-IDF vectorization of:
  - Movie overviews
  - Genres
  - Keywords
- Weighted cosine similarity across multiple text features
- Rating score adjustment based on:
  - Average rating
  - Number of votes (confidence weighting)
- Interactive command-line interface
- Modular project structure with separate preprocessing and model stages

---

How It Works

1. Data Cleaning & Preprocessing
- Raw movie metadata is cleaned and split into:
  - A dataset for TF-IDF computation
  - A dataset for displaying movie information
- Missing, empty, and invalid values are handled explicitly.

2. Text Vectorization
- TF-IDF vectorizers are trained separately for:
  - `keywords`
  - `genres`
  - `overview`
- This allows each text feature to be weighted independently.

3. Similarity Calculation
- Cosine similarity is computed between the selected movie and all others.
- Similarity scores from each text feature are combined using predefined weights.

4. Rating Adjustment
- Each movie’s similarity score is adjusted using:
  - Its average rating
  - A confidence weight derived from the number of votes

5. Recommendation Output
- The top *N* most similar movies are returned.
- Additional metadata (tagline, runtime, cast, etc.) is displayed where available.

---

How to run

- Clone the repository
- Install dependencies specified in requirements.txt
- Run the recommendation system (model_3.py)
- Follow the prompts by the UI (enter movie title, enter number of recommendations etc)

---

Example output

Title: Pulp Fiction (1994)
Tagline: Just because you are a character doesn't mean you have character.
Runtime: 154 minutes
Overview: A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper. Their adventures unfurl in three stories that ingeniously trip back and forth in time.
Directors: Quentin Tarantino
Cast: John Travolta, Samuel L. Jackson, Uma Thurman, Bruce Willis, Ving Rhames, Harvey Keitel, Eric Stoltz, Tim Roth, Amanda Plummer, Maria de Medeiros

---

Dataset

The project uses a large movie dataset derived from TMDb/IMDb sources.
All data is used for educational and non-commercial purposes.

Dataset Availability

Due to file size and licensing restrictions, the raw and processed datasets are not included in this repository.
To run the project locally:
-  Obtain a compatible movie metadata dataset (eg TMDB/IMDB)
- Place the raw file in data/raw/
- Run the preprocessing scripts in scripts/preprocessing/

---

Author (Harvey Appleyard)

Built as a personal learning and portfolio project to explore NLP, recommendation systems, and applied machine learning concepts.
