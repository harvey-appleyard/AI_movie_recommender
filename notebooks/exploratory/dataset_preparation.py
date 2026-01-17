import pandas as pd

df = pd.read_csv('../../data/raw/TMDB_IMDB_Movies_Dataset.csv')

#Only these 4 columns
cleaned = df[['id', 'title', 'genres', 'overview']].copy()

#Drop row if no id, title or overview
cleaned = cleaned.dropna(subset=['id', 'title', 'overview'])

#Ensure text columns are strings (prevents float / NaN issues)
for column in ['title', 'genres', 'overview']:
    cleaned[column] = cleaned[column].astype(str)

#Normalise text: lowercase and strip whitespace
cleaned['title'] = cleaned['title'].str.lower().str.strip()
cleaned['genres'] = cleaned['genres'].str.lower().str.strip()
cleaned['overview'] = cleaned['overview'].str.lower().str.strip()

#drop rows with duplicated ids
cleaned = cleaned.drop_duplicates(subset=['id'])

#Reset index to align with TF-IDF matrix
cleaned = cleaned.reset_index(drop=True)


#Save
cleaned.to_csv('../../data/processed/Cleaned_Movies_Dataset.csv', index=False)

