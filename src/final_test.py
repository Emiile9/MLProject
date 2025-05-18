from xgboost import XGBClassifier, plot_importance
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from movies_preprocessing import full_processing
from evaluation import get_final_ypred, compute_topk_accuracy

def get_xgb_importance(model, plot_folder, max_num_features=20):
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=max_num_features, importance_type='gain', show_values=False)
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plot_path = os.path.join(plot_folder, 'importanceXGB.png') 
    plt.savefig(plot_path)
    plt.close()

def plot_confusion(y_true, y_pred, plot_folder):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Winner", "Non-winner"])
    disp.plot(cmap="Blues", values_format='d')
    plot_path = os.path.join(plot_folder, 'confusion_matrix_XGB.png') 
    plt.savefig(plot_path)
    plt.close()

def final_results(model, X_test_aligned):
    y_prob = model.predict_proba(X_test_aligned)[:, 1]
    X_test_aligned.to_csv("X_test", index=False)
    y_pred = get_final_ypred(y_prob, X_test_aligned)
    return y_pred, y_prob 


plot_folder = os.path.join('..', 'plots')
training = pd.read_csv("../data/training.csv")
testing = pd.read_csv("../data/testing.csv")

features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X_train = training[features]
y_train = training["winner"]
X_test = testing[features]
y_test = testing["winner"]

X_train_processed = full_processing(X_train, "median")
X_test_processed = full_processing(X_test, "median")
#We need to handle genres appearing in the train set but not in the test set
X_test_aligned = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

XGB = XGBClassifier(n_estimators = 200, max_depth = 3, reg_lambda = 5, learning_rate = 0.01, use_label_encoder=False, eval_metric='logloss', verbosity=0,)
XGB.fit(X_train_processed, y_train)

y_pred, y_prob = final_results(XGB, X_test_aligned)
top1, top3 = compute_topk_accuracy(XGB, X_test_aligned, y_test, 3)
print(top1, top3)

plot_confusion(y_test, y_pred, plot_folder)

get_xgb_importance(XGB, plot_folder)