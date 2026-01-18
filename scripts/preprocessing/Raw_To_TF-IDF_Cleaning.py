import pandas as pd

#Load raw dataset
df = pd.read_csv('../../data/raw/Original_Raw_Movie_Dataset.csv')

#Only these 4 columns
cleaned = df[['id', 'title', 'genres', 'overview']].copy()

#Drop row if no id, title or overview
cleaned.dropna(subset=['id', 'title', 'overview'], inplace=True)

#Ensure text columns are strings (prevents float / NaN issues)
for column in ['title', 'genres', 'overview']:
    cleaned[column] = cleaned[column].astype(str)

#Strip leading and trailing whitespace
cleaned['title'] = cleaned['title'].str.strip()
cleaned['genres'] = cleaned['genres'].str.strip()
cleaned['overview'] = cleaned['overview'].str.strip()

#drop rows with duplicated ids
cleaned.drop_duplicates(subset=['id'], inplace=True)

#Reset index to align with TF-IDF matrix
cleaned.reset_index(drop=True, inplace=True)

#Save
cleaned.to_csv('../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv', index=False)
