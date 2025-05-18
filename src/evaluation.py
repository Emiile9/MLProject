from itertools import product
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
    
def get_n_accuracies_test(model, df, X, y, nb_of_runs=100):
    cpt_top1 = 0
    cpt_top3 = 0
    top1_acc = []
    top3_acc = []
    for i in range(nb_of_runs):
        X_train, X_test, y_train, y_test = train_test_split_perso(df, X, y, 0.2)
        model.fit(X_train, y_train)
        top1, top3 = compute_topk_accuracy(model, X_test, y_test,3)
        top1_acc.append(top1)
        top3_acc.append(top3)
    return top1_acc, top3_acc

def get_n_accuracies_train(model, df, X, y, nb_of_runs=100):
    cpt_top1 = 0
    cpt_top3 = 0
    top1_acc = []
    top3_acc = []
    for i in range(nb_of_runs):
        X_train, X_test, y_train, y_test = train_test_split_perso(df, X, y, 0.2)
        model.fit(X_train, y_train)
        top1, top3 = compute_topk_accuracy(model, X_train, y_train,3)
        top1_acc.append(top1)
        top3_acc.append(top3)
    return top1_acc, top3_acc

def custom_gridsearch(model_type, df, X, y, param_grid, nb_runs):
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    scores = []
    for params in all_combinations:
        model = model_type(**params)
        top1 = get_n_accuracies(model, df, X, y, nb_runs)[0]
        scores.append({**params, "top1": top1})
    return pd.DataFrame(scores)

'''
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
'''