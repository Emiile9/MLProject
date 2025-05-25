from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np

def genre_encoding(df):
    df["genres_list"] = df["genres"].str.split(",")
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(df["genres_list"])
    genre_df = pd.DataFrame(encoded_genres, columns=mlb.classes_, index=df.index)
    df_encoded = pd.concat([df.drop(columns=["genres", "genres_list"]), genre_df], axis=1)
    return df_encoded


def fill_budget(row, df, remplacement):
    if pd.notnull(row['budget']):
        return row['budget']
    min_year = row['year_film'] - 1
    max_year = row['year_film'] + 1
    if remplacement  == "mean":
        mean_budget = df[(df['year_film'] >= min_year) & 
                       (df['year_film'] <= max_year) & 
                       (df['budget'].notna())]['budget'].mean()
        return mean_budget
    else:
        median_budget = df[(df['year_film'] >= min_year) & 
                       (df['year_film'] <= max_year) & 
                       (df['budget'].notna())]['budget'].median()
        return median_budget

def budget_preprocessing(df, replacement):
    df_filled = df.copy()
    df_filled['budget'] = df.apply(lambda row: fill_budget(row, df_filled, replacement), axis=1) 
    df_filled['budget'] = np.log1p(df_filled['budget'])
    return df_filled

def Scaling(df):
    columns_to_scale = ["budget", "averageRating", "runtimeMinutes", "nb_actor_won_before"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df_scaled

def full_processing(df, budget_replacement):
    df_genres = genre_encoding(df)
    df_budget = budget_preprocessing(df_genres, budget_replacement)
    df_scaled = Scaling(df_budget)
    return df_scaled
