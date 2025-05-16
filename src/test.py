from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
from movies_preprocessing import full_processing

df = pd.read_csv('../data/final_data.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
X_processed = X_processed.drop(columns=['year_film'])
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.1)
logReg = LogisticRegression(class_weight='balanced', penalty = 'l2')
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
y_prob = logReg.predict_proba(X_test)[:, 1]

# 5. Evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))