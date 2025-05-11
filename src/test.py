from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import polars as pl
df = pl.read_csv('final_data.csv')
features = ['dir_won_before', 'nb_actor_won_before', 'averageRating', 'won_bafta', 'won_gg_drama', 'won_gg_comedy']
df = df.filter(pl.col("averageRating").is_not_null())

target = 'winner'
X = df[features]
y = df [target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
logReg = LogisticRegression(class_weight='balanced', penalty = 'l2')
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
y_prob = logReg.predict_proba(X_test)[:, 1]

# 5. Evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))