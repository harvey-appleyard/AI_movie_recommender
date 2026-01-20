import pandas as pd

#Load raw dataset
df = pd.read_csv('../../data/raw/Original_Raw_Movie_Dataset.csv')

#Only these columns
cleaned = df[['id', 'title', 'release_date', 'runtime', 'tagline', 'overview', 'directors', 'cast']].copy()

#Drop row if no id, title or overview
cleaned.dropna(subset=['id', 'title', 'overview'], inplace=True)

#Ensure columns are strings (prevents float / NaN issues) and strip
for column in cleaned.columns:
    cleaned[column] = cleaned[column].astype(str)
    cleaned[column] = cleaned[column].str.strip()

#Drop rows with duplicated ids
cleaned.drop_duplicates(subset=['id'], inplace=True)

#Align indexes
cleaned.reset_index(drop=True, inplace=True)

#Save
cleaned.to_csv('../../data/processed/Movie_Info_Dataset.csv', index=False)
