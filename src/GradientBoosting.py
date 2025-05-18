from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso
from evaluation import get_n_accuracies
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV

df = pd.read_csv('../data/training.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)

model = XGBClassifier()
model.fit(X_train, y_train)
top1, top3 =  get_n_accuracies(model, df, X_processed, y, 100)
print(np.mean(top1))