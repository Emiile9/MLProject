from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
from movies_preprocessing import full_processing
from sklearn.model_selection import GroupShuffleSplit

#For each year we select the movie the highest probability of winning
def get_final_ypred(model, X_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_df = pd.DataFrame({
        "year_film": X_test["year_film"],
        "probs": y_prob
    })
    pred_top1 = []
    for year, group in prob_df.groupby("year_film"):
        idx = group["probs"].idxmax()
        pred_top1.append(idx)
    y_pred = pd.Series(0, index=prob_df.index)
    y_pred.loc[pred_top1] = 1
    return y_pred

#For manual checking only
def concatenate_results(X_test, y_test, y_pred):
    results = X_test.copy()
    results["true_winner"] = y_test.values
    results["pred_winner"] = y_pred.values
    return results

df = pd.read_csv('../data/final_data.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']

groups = df['year_film']

gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
train_idx, test_idx = next(gss.split(X_processed, y, groups))

X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

logReg = LogisticRegression(class_weight='balanced', penalty = 'l2')
logReg.fit(X_train, y_train)

y_pred = get_final_ypred(logReg, X_test)
results = concatenate_results(X_test, y_test, y_pred)
results.to_csv("results.csv", index=False)
# 5. Evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))