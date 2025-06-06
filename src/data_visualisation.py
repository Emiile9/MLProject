import matplotlib.pyplot as plt
import polars as pl 
import pandas as pd
import seaborn as sns 
import os

df = pl.read_csv('../data/final_oscar_data.csv')
winners_df = df.filter(pl.col("winner") == 1)
nominees_df = df.filter(pl.col("winner") == 0)

#Checking Ratings Distribution 
overall_ratings = df['averageRating']
winners_ratings = winners_df['averageRating']
nominees_ratings = nominees_df['averageRating']

#distribution plot
plt.figure(figsize=(8, 5))
sns.kdeplot(nominees_ratings, color='red', label='Nominees Distribution', fill=False)
sns.kdeplot(winners_ratings, color='green', label='Winners Distribution', fill=False)
plt.title('Rating Distribution for Nominees and Winners', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'ratings_plot.png') 
plt.savefig(plot_path)
plt.close()

#Same for budget 
overall_budget = df['budget']
winners_budget = winners_df['budget']
nominees_budget = nominees_df['budget']

#distribution plot
plt.figure(figsize=(8, 5))
sns.kdeplot(nominees_budget, color='red', label='Nominees Distribution', fill=False)
sns.kdeplot(winners_budget, color='green', label='Winners Distribution', fill=False)
plt.title('Budget Distribution for Nominees and Winners', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plot_path = os.path.join(plot_folder, 'budget_plot.png') 
plt.savefig(plot_path)
plt.close()

#Same for runtime
overall_runtimeMinutes = df['runtimeMinutes']
winners_runtimeMinutes = winners_df['runtimeMinutes']
nominees_runtimeMinutes = nominees_df['runtimeMinutes']

#distribution plot
plt.figure(figsize=(8, 5))
sns.kdeplot(nominees_runtimeMinutes, color='red', label='Nominees Distribution', fill=False)
sns.kdeplot(winners_runtimeMinutes, color='green', label='Winners Distribution', fill=False)
plt.title('Runtime Distribution for Nominees and Winners', fontsize=14)
plt.xlabel('Runtime', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plot_path = os.path.join(plot_folder, 'runtimeMinutes_plot.png') 
plt.savefig(plot_path)
plt.close()

#Bar chart for genres comparing winners and nominees
def get_genre_counts(data):
    exploded = data.assign(genres=data['genres'].str.split(',')).explode('genres')
    return exploded['genres'].value_counts()

nominees_pandas = nominees_df.to_pandas()
winners_pandas = winners_df.to_pandas()
nominee_genres = get_genre_counts(nominees_pandas)
winner_genres = get_genre_counts(winners_pandas)


nominee_percent = nominee_genres / len(nominees_pandas) * 100
winner_percent = winner_genres / len(winners_pandas) * 100

df_genres = pd.DataFrame({
    'Nominees (%)': nominee_percent,
    'Winners (%)': winner_percent
}).fillna(0)


df_genres['Average'] = (df_genres['Nominees (%)'] + df_genres['Winners (%)']) / 2
df_genres = df_genres.sort_values('Average', ascending=True).drop(columns='Average')

ax = df_genres.plot(kind='barh', figsize=(10, 12), width=0.7)
plt.title('Genre Distribution (%): Nominees vs. Winners')
plt.xlabel('Percentage of Films')
plt.ylabel('Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'genres_plot.png') 
plt.savefig(plot_path)
plt.close()

#HeatMap for other awards won
df_pandas = df.to_pandas()
combo = df_pandas.groupby(['won_gg_drama', 'won_gg_comedy', 'won_bafta'])['winner'].mean().unstack(fill_value=0)
sns.heatmap(combo, annot=True, cmap='YlGnBu')
plt.title("Oscar Win Rate by Other Award Wins")
plot_path = os.path.join(plot_folder, 'awards_plot.png') 
plt.savefig(plot_path)
plt.close()