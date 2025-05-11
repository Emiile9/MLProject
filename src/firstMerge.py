import pandas as pd
import polars as pl
import re

def clean_title(title):
    return re.sub(r'[^A-Za-z0-9]', '', str(title).lower())

def filter_closest_match(group, nominee_year):
    # If there's an exact match, return it
    exact_match = group[group['startYear'] == nominee_year]
    if not exact_match.empty:
        return exact_match.iloc[0]  # Keep the exact match

    # Otherwise, return the closest available match
    return group.iloc[(group['startYear'] - nominee_year).abs().argmin()]  

df = pd.read_csv("oscars.csv", encoding='latin-1')
features = ['year_film', 'canon_category', 'film', 'winner']
df = df[features]
nominees = df[df['canon_category'] == 'BEST PICTURE']
nominees['film'] = nominees['film'].apply(clean_title)
nominees['year_film'] = nominees['year_film'].astype(int)


columns_to_use = ['tconst', 'titleType', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres']
imdbBasics = pd.read_csv('title.basics.tsv', sep='\t', usecols = columns_to_use)
imdbBasics = imdbBasics[imdbBasics['titleType'] == 'movie']
imdbBasics['primaryTitle'] = imdbBasics['primaryTitle'].apply(clean_title)
imdbBasics['startYear'] = pd.to_numeric(imdbBasics['startYear'], errors='coerce')
# Step 1: Sort by title and absolute year difference
imdbBasics_sorted = imdbBasics.sort_values(by=['primaryTitle', 'startYear'])



# Step 3: Apply function per title group
filtered_rows = []
for _, nominee_row in nominees.iterrows():
    title = nominee_row['film']
    nominee_year = nominee_row['year_film']
    
    # Get all IMDb matches for this title
    matches = imdbBasics_sorted[imdbBasics_sorted['primaryTitle'] == title]
    
    if not matches.empty:
        best_match = filter_closest_match(matches, nominee_year)
        filtered_rows.append(best_match)

# Step 4: Convert back to DataFrame
imdbBasics = pd.DataFrame(filtered_rows)

merged_df = nominees.merge(imdbBasics,left_on='film', right_on='primaryTitle', how='left')
merged_df = merged_df[(merged_df['year_film'] >= merged_df['startYear'] - 2) & (merged_df['year_film'] <= merged_df['startYear'] + 2)]
missing_rows = nominees[~nominees['film'].isin(merged_df['primaryTitle'])]
print(missing_rows[['film', 'year_film']])
merged_df = merged_df.drop_duplicates(subset=['film', 'year_film'])


nominees_pairs = set(nominees[['film', 'year_film']].itertuples(index=False, name=None))
merged_pairs = set(merged_df[['film', 'year_film']].itertuples(index=False, name=None))

# Find missing (title, year) pairs
missing_pairs = nominees_pairs - merged_pairs

# Print missing movies
print(f"Missing movies: {len(missing_pairs)}")
for title, year in missing_pairs:
    print(f"Missing: {title} ({year})")
    
dfpolars = pl.from_pandas(merged_df)
print(dfpolars)
    

duplicate_titles = merged_df['primaryTitle'].value_counts()

# Filter only titles that appear more than once
duplicate_titles = duplicate_titles[duplicate_titles > 1]

# Display them
duplicate_rows = merged_df[merged_df['film'].isin(duplicate_titles.index)]

# Display duplicate rows
print(duplicate_rows[['year_film', 'film']])
#%%
missing_data = {key : [] for key in dfpolars.columns}
for title, year in missing_pairs:
    print("What's the imdb title of ", title, year)
    imdbtitle = str(input())
    for key in nominees.columns:
        if key in missing_data.keys():
            missing_data[key].append(nominees[nominees['film'] == title][key].values[0])
    for key in imdbBasics.columns:
        if key in missing_data.keys():
            filtered = imdbBasics_sorted[
                (imdbBasics_sorted['primaryTitle'] == clean_title(imdbtitle)) &
                (imdbBasics_sorted['startYear'].astype(float).between(year - 1,year + 1))
            ]
            value = filtered[key].values[0]
            missing_data[key].append(value)
#%%
for key in missing_data.keys():
    print("what's the ", key)
    value = input()
    missing_data[key].append(value)
#%%
missingdf = pd.DataFrame(missing_data)
fulldf = pd.concat([merged_df, missingdf], ignore_index=True)
finaldf = fulldf[['tconst', 'film', 'year_film', 'runtimeMinutes', 'genres', 'winner']]
finaldf.to_csv('firstmerge.csv', index = False)
