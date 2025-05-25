import numpy as np
import polars as pl
df = pl.read_csv('../data/final_oscar_data.csv')

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

df_clean_budgets = df.filter(df["budget"].is_not_null())
winners_budget_df = df_clean_budgets.filter(pl.col("winner") == 1)
nominees_budget_df = df_clean_budgets.filter(pl.col("winner") == 0)

overall_budget = df_clean_budgets["budget"].to_list()
winners_budget = winners_budget_df["budget"].to_list()
nominees_budget = nominees_budget_df["budget"].to_list()
budget_data = {
    'Group': ['Winners', 'Non-Winners', 'All Budgets'],
    'Mean': [np.mean(winners_budget), np.mean(nominees_budget), np.mean(overall_budget)],
    'Median': [np.median(winners_budget), np.median(nominees_budget), np.median(overall_budget)]
}

print(pl.DataFrame(budget_data))

df_clean_runtimeMinutess = df.filter(df["runtimeMinutes"].is_not_null())
winners_runtimeMinutes_df = df_clean_runtimeMinutess.filter(pl.col("winner") == 1)
nominees_runtimeMinutes_df = df_clean_runtimeMinutess.filter(pl.col("winner") == 0)

overall_runtimeMinutes = df_clean_runtimeMinutess["runtimeMinutes"].to_list()
winners_runtimeMinutes = winners_runtimeMinutes_df["runtimeMinutes"].to_list()
nominees_runtimeMinutes = nominees_runtimeMinutes_df["runtimeMinutes"].to_list()
runtimeMinutes_data = {
    'Group': ['Winners', 'Non-Winners', 'All Runtimes'],
    'Mean': [np.mean(winners_runtimeMinutes), np.mean(nominees_runtimeMinutes), np.mean(overall_runtimeMinutes)],
    'Median': [np.median(winners_runtimeMinutes), np.median(nominees_runtimeMinutes), np.median(overall_runtimeMinutes)]
}

print(pl.DataFrame(runtimeMinutes_data))