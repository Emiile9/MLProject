import numpy as np
import polars as pl
df = pl.read_csv('../data/final_data.csv')

df_clean_ratings = df.filter(df["averageRating"].is_not_null())
winners_df = df_clean_ratings.filter(pl.col("winner") == 1)
nominees_df = df_clean_ratings.filter(pl.col("winner") == 0)

overall_ratings = df_clean_ratings['averageRating'].to_list()
winners_ratings = winners_df['averageRating'].to_list()
nominees_ratings = nominees_df['averageRating'].to_list()

ratings_data = {
    'Group': ['Winners', 'Non-Winners', 'All Ratings'],
    'Mean': [np.mean(winners_ratings), np.mean(nominees_ratings), np.mean(overall_ratings)],
    'Median': [np.median(winners_ratings), np.median(nominees_ratings), np.median(overall_ratings)]
}

print(pl.DataFrame(ratings_data))