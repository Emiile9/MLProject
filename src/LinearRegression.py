from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso
from sklearn.model_selection import GridSearchCV, GroupKFold
from evaluation import compute_topk_accuracy, get_n_accuracies, get_final_ypred, custom_gridsearch

df = pd.read_csv('../data/training.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)
#Baseline Logistic Regression
log_reg_raw = LogisticRegression(max_iter=1000, class_weight='balanced', solver = 'liblinear')
log_reg_raw.fit(X_train, y_train)

log_reg_regu = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000)

top1_scores_mean = []
top3_scores_mean = []
top1_scores_std = []
top3_scores_std = []
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
for c in Cs:
    log_reg_c = LogisticRegression(penalty='l2', C = c, max_iter=1000, solver='liblinear', class_weight='balanced')
    top1, top3 = get_n_accuracies(log_reg_c, df, X_processed, y, 1000)
    top1_scores_mean.append(np.mean(top1))
    top3_scores_mean.append(np.mean(top3))
    top1_scores_std.append(np.std(top1))
    top3_scores_std.append(np.std(top3))

print(top3_scores_mean)
print(top1_scores_mean)
plt.errorbar(np.log(Cs), top1_scores_mean, yerr=top1_scores_std, fmt='-o', capsize=5, label="Top-1 Score")
plt.errorbar(np.log(Cs), top3_scores_mean, yerr=top3_scores_std, fmt='-o', capsize=5, label="Top-3 Score")


plt.xlabel("Value of log(C)")
plt.ylabel("Accuracy scores")
plt.title("Accuracy Scores for different values of C")
plt.legend()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'C_choice_logReg.png') 
plt.savefig(plot_path)
plt.close()
