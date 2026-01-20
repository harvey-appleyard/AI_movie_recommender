import pandas as pd

#Load raw dataset
df = pd.read_csv('../../data/raw/Original_Raw_Movie_Dataset.csv')

#Only these 4 columns
cleaned = df[['id', 'title', 'genres', 'overview']].copy()

#Drop row if no id, title or overview
cleaned.dropna(subset=['id', 'title', 'overview'], inplace=True)

#Ensure columns are strings (prevents float / NaN issues) and strip
for column in cleaned.columns:
    cleaned[column] = cleaned[column].astype(str)
    cleaned[column] = cleaned[column].str.strip()

#drop rows with duplicated ids
cleaned.drop_duplicates(subset=['id'], inplace=True)

#Reset index to align with TF-IDF matrix
cleaned.reset_index(drop=True, inplace=True)

#Save
cleaned.to_csv('../../data/processed/Cleaned_For_TF-IDF_Movie_Dataset.csv', index=False)
