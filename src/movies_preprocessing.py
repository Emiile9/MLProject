from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
import pandas as pd

def genre_encoding(df):
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(df["genres"])
    genre_df = pd.DataFrame(encoded_genres, columns=mlb.classes_, index=df.index)
    df_encoded = pd.concat([df.drop(columns=["genres"]), genre_df], axis=1)
    return df_encoded


df = pd.read_csv('../data/final_data.csv')
encoded = genre_encoding(df)
encoded.to_csv('encoded.csv', index=False)
print(encoded)