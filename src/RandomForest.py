from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso
from evaluation import get_n_accuracies_test, get_n_accuracies_train

df = pd.read_csv('../data/training.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)
#Baseline Random Forest
RF_base = RandomForestClassifier(class_weight='balanced')
RF_base.fit(X_train, y_train)
top1, top3 = get_n_accuracies_test(RF_base, df, X_processed, y, 100)
print(np.mean(top1), np.mean(top3))

depths = [1,3, 5, None]
n_trees = [10, 50, 100, 200, 1000, 2000]
results_training = []
results_validation = []
for d in depths:
    for n in n_trees:
        model = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight='balanced')
        
        # Compute training accuracy
        top1_train = get_n_accuracies_train(model, df, X_processed, y, 100)[0]
        results_training.append({
            'max_depth': d,
            'n_estimators': n,
            'top1_accuracy': np.mean(top1_train)
        })
        
        # Compute validation accuracy
        top1_val = get_n_accuracies_test(model, df, X_processed, y, 100)[0]
        results_validation.append({
            'max_depth': d,
            'n_estimators': n,
            'top1_accuracy': np.mean(top1_val)
        })



df_train = pd.DataFrame(results_training)
df_val = pd.DataFrame(results_validation)
df_val["max_depth"] = df_val["max_depth"].astype(str)
df_train["max_depth"] = df_train["max_depth"].astype(str)

train_pivot = df_train.pivot(index='max_depth', columns='n_estimators', values='top1_accuracy')
val_pivot = df_val.pivot(index='max_depth', columns='n_estimators', values='top1_accuracy')

#Plot for validation 
plt.figure(figsize=(10, 6))
sns.heatmap(val_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Top-1 Accuracy by max_depth and n_estimators")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Max Depth")
plt.tight_layout()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'Top1_valid_RF.png') 
plt.savefig(plot_path)
plt.close()

#Plot for validation 
plt.figure(figsize=(10, 6))
sns.heatmap(train_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Top-1 Accuracy by max_depth and n_estimators")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Max Depth")
plt.tight_layout()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'Top1_train_RF.png') 
plt.savefig(plot_path)
plt.close()