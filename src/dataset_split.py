import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Only used once when splitting the dataset in two
def split_dataset_by_year(df, test_years):
    years_sorted = sorted(df["year_film"].unique())
    test_years_list = years_sorted[-test_years:]
    df_train = df[~df["year_film"].isin(test_years_list)].copy()
    df_test = df[df["year_film"].isin(test_years_list)].copy()
    return df_train, df_test
df = pd.read_csv("../data/final_oscar_data.csv")
df_train, df_test = split_dataset_by_year(df, 15)
df_train.to_csv("../data/training.csv", index=False)
df_test.to_csv("../data/testing.csv", index = False)
def train_test_split_perso(df, X, y, test_size):
    groups = df['year_film']
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test