import pandas as pd

def split_dataset_by_year(df, test_years):
    years_sorted = sorted(df["year_film"].unique())
    test_years_list = years_sorted[-test_years:]
    df_train = df[~df["year_film"].isin(test_years_list)].copy()
    df_test = df[df["year_film"].isin(test_years_list)].copy()
    return df_train, df_test

df = pd.read_csv('../data/final_data.csv')
training, testing = split_dataset_by_year(df, 15)
training.to_csv('../data/training.csv')
testing.to_csv('../data/testing.csv')
