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

def budget_processing(df, replacement_value):
    df['budget_filled'] = df['budget']
    if replacement_value == 'mean':
        df['budget_filled'] = df.groupby('year_film')['budget_filled'].transform(lambda x: x.fillna(x.mean()))
    else:
        df['budget_filled'] = df.groupby('year_film')['budget_filled'].transform(lambda x: x.fillna(x.median()))
    df['budget_log'] = np.log1p(df['budget_filled'])
    df.drop(columns=["budget", "budget_filled"])
    return df