from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso

#For each year we select the movie the highest probability of winning
def get_final_ypred(y_prob, X_test):
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
def concatenate_results(X_test, y_test, y_pred, y_prob):
    results = X_test.copy()
    results["true_winner"] = y_test.values
    results["pred_winner"] = y_pred.values
    results["proba"] = y_prob
    return results

def compute_topk_accuracy(model, X_test, y_test,k):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = get_final_ypred(y_prob, X_test)
    results = concatenate_results(X_test, y_test, y_pred, y_prob)
    top1_correct = 0
    topk_correct = 0
    total_years = results["year_film"].nunique()
    for _, group in results.groupby("year_film"):
        group_sorted = group.sort_values(by="proba", ascending=False)
        top1_correct += int(group_sorted.iloc[0]["true_winner"] == 1) #Check if index of highest proba is same as index of true winner 
        topk_correct += int(group_sorted.head(k)["true_winner"].sum() > 0) #Check if any of the k highest proba is the real winner
    top1_acc = top1_correct/total_years
    topk_acc = topk_correct/total_years
    return top1_acc, topk_acc
    
def get_n_accuracies(model, df, X, y, nb_of_runs=100):
    cpt_top1 = 0
    cpt_top3 = 0
    for i in range(nb_of_runs):
        X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = get_final_ypred(y_prob, X_test)
        results = concatenate_results(X_test, y_test, y_pred, y_prob)
        top1, top3 = compute_topk_accuracy(results, 3)
        cpt_top1 += top1
        cpt_top3 += top3
    avg_top1 = cpt_top1/nb_of_runs
    avg_top3 = cpt_top3/nb_of_runs
    return avg_top1, avg_top3

df = pd.read_csv('../data/training.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)
logReg = LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l1', solver='liblinear')
logReg.fit(X_train, y_train)


'''
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
'''