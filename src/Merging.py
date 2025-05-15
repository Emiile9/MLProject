import pandas as pd
import re
import budget
def clean_names(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name).lower())

def clean_title(title):
    return re.sub(r'[^A-Za-z0-9]', '', str(title).lower())

names = pd.read_csv('name.basics.tsv', sep='\t', usecols=['nconst', 'primaryName'])
#We merge the first built dataframe with other imdb dataframes using the tconst value
#%%
#Adding the ratings
columns_to_use = ['tconst', 'averageRating']
firstmerge = pd.read_csv('../data/firstmerge.csv')
ratings = pd.read_csv('title.ratings.tsv', sep='\t', usecols = columns_to_use)
with_ratings = firstmerge.merge(ratings,on='tconst',how='left')
#%%
#Adding the directors
directors = pd.read_csv('title.crew.tsv', sep='\t', usecols=['tconst', 'directors'])
with_directors = with_ratings.merge(directors, on='tconst', how='left')

#When a film has multiple directors we split the row in one row per director
with_directors['directors'] = with_directors['directors'].str.split(',')
with_directors = with_directors.explode('directors')
with_names = with_directors.merge(names, left_on='directors', right_on='nconst', how='left')
#Then regroup to get one row per movie
grouped = with_names.groupby('tconst').agg({
    'film': 'first',
    'year_film': 'first',
    'runtimeMinutes': 'first',
    'genres': 'first',
    'averageRating': 'first',
    'winner': 'first',
    'directors': 'first',  
    'nconst': lambda x: ','.join(x.dropna().unique()),  
    'primaryName': lambda names: ', '.join(names.dropna().unique())  
}).reset_index()
directors_names = grouped[['tconst', 'film', 'year_film', 'genres', 'averageRating', 'primaryName', 'winner']]
directors_names['primaryName'] = directors_names['primaryName'].apply(clean_names)
#%%
#Adding a column checking if director has won an oscar before
df = pd.read_csv("oscars.csv", encoding='latin-1', usecols = ['year_film', 'name', 'canon_category', 'winner'])
winners = df[(df['canon_category'] == 'DIRECTING') & (df['winner'] == True)][['year_film', 'name']]
winners['name'] = winners['name'].apply(clean_names)
winners.rename(columns={'year_film': 'year_won'}, inplace=True)

with_winners = directors_names.merge(winners, left_on='primaryName', right_on='name', how='left')
with_winners['dir_won_before'] = (with_winners['year_won'] <= with_winners['year_film']).astype(int)
with_winners = with_winners.drop_duplicates(subset='tconst')
with_winners = with_winners[['tconst', 'film', 'year_film', 'genres', 'averageRating', 'dir_won_before', 'winner']]
with_winners.to_csv('with_dir.csv', index = False)

#%%
#Adding the budget using SPARQL
imdb_ids = with_winners['tconst'].unique().tolist()
budget_df = budget.get_budgets_from_wikidata(imdb_ids)

with_budget = with_winners.merge(budget_df, on = 'tconst', how = 'left')

#%%
#Adding the list of actors for each movie in the dataset
principals = pd.read_csv('title.principals.tsv', sep = '\t', usecols=['tconst', 'nconst', 'category'])
actors = principals[(principals['category'] == 'actor')]
actresses = principals[(principals['category'] == 'actress')]
full_actors = pd.concat([actors, actresses], ignore_index = False)
actors_names = full_actors.merge(names, on = 'nconst', how='left')
grouped = df.groupby('tconst')['names'].agg(lambda x: list(set(x))).reset_index()
with_actors = with_budget.merge(grouped, on = 'tconst', how='left')
#%%
exploded = with_actors.explode('primaryName')
categories = ['ACTOR IN A LEADING ROLE', 'ACTRESS IN A LEADING ROLE', 'ACTOR IN A SUPPORTING ROLE', 'ACTRESS IN A SUPPORTING ROLE']
actor_winners = df[(df['canon_category'].isin(categories)) & (df['winner'] == True)][['year_film', 'name']]
actor_winners.rename(columns={'year_film': 'year_won'}, inplace=True)
merged = exploded.merge(actor_winners, left_on='primaryName', right_on='name', how='left')
merged['won_before'] = merged['year_won'] < merged['year_film']
actor_win_counts = (
    merged.groupby('tconst')['won_before']
    .sum()
    .astype(int)
    .reset_index()
    .rename(columns={'won_before': 'nb_actor_won_before'})
)
#%%
actorsdf = with_budget.merge(actor_win_counts, on='tconst', how='left')
actorsdf.to_csv('finaldata.csv', index = False)
#%%
actorsdf = pd.read_csv('finaldata.csv', usecols = ['tconst', 'film', 'year_film_x', 'genres', 'averageRating',
       'dir_won_before', 'winner', 'budget', 'nb_actor_won_before'])
actorsdf.rename(columns={'year_film_x': 'year_film'}, inplace=True)

gg_drama = pd.read_csv('Golden_Globes_BestDrama.csv')
gg_comedy = pd.read_csv('Golden_Globes_BestComedy.csv')
bafta = pd.read_csv('BAFTA.csv')

bafta['film'] = bafta['film'].apply(clean_title)
gg_drama['film'] = gg_drama['film'].apply(clean_title)
gg_comedy['film'] = gg_comedy['film'].apply(clean_title)

awards = {'film' : [], 'year_film' : [], 'won_bafta' : [], 'won_gg_drama' : [], 'won_gg_comedy' : []}
films = actorsdf[['film', 'year_film']].to_numpy()
for film in films:
    title, year = film[0], film[1]
    filteredbafta = bafta[
        (bafta['film'] == title )&
        (bafta['year'].astype(float).between(year - 1,year + 1))
    ]
    filteredggD = gg_drama[
        (gg_drama['film'] == title )&
        (gg_drama['year'].astype(float).between(year - 1,year + 1))
    ]
    filteredggC = gg_comedy[
        (gg_comedy['film'] == title )&
        (gg_comedy['year'].astype(float).between(year - 1,year + 1))
    ]
    awards['film'].append(title)
    awards['year_film'].append(year)
    awards['won_bafta'].append(len(filteredbafta))
    awards['won_gg_drama'].append(len(filteredggD))
    awards['won_gg_comedy'].append(len(filteredggC))

awardsdf = pd.DataFrame(awards)

final = actorsdf.merge(awardsdf, on = ['film', 'year_film'], how='left')
final['winner'] = final['winner'].astype(int)
final = final[['tconst', 'film', 'year_film', 'genres', 'averageRating',
       'dir_won_before', 'budget', 'nb_actor_won_before',
       'won_bafta', 'won_gg_drama', 'won_gg_comedy', 'runtimeMinutes', 'winner']]

final.drop_duplicates(subset='tconst', inplace=True)
final.to_csv('final_data.csv', index = False)

