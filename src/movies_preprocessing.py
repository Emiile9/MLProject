from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
import pandas as pd

def genre_encoding(df):
    df["genres_list"] = df["genres"].str.split(",")
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(df["genres_list"])
    genre_df = pd.DataFrame(encoded_genres, columns=mlb.classes_, index=df.index)
    df_encoded = pd.concat([df.drop(columns=["genres", "genres_list"]), genre_df], axis=1)
    return df_encoded

