from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso
from evaluation import get_n_accuracies_test, get_n_accuracies_train

df = pd.read_csv('../data/training.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)

model = XGBClassifier(scale_pos_weight = 0.1)
model.fit(X_train, y_train)
top1, top3 =  get_n_accuracies_test(model, df, X_processed, y, 100)
print(np.mean(top1), np.mean(top3))

'''
# Define hyperparameter values
depths = [3, 5, 10]
n_trees = [100, 200, 1000]
learning_rates = [0.01, 0.1]
scale_pos_weights = [1, 5]
reg_lambdas = [0, 1, 10]  # L2 regularization

results_training = []
results_validation = []

for d in depths:
    for n in n_trees:
        for lr in learning_rates:
            for spw in scale_pos_weights:
                for reg in reg_lambdas:
                    model = XGBClassifier(
                        n_estimators=n,
                        max_depth=d,
                        learning_rate=lr,
                        scale_pos_weight=spw,
                        reg_lambda=reg,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=0,
                    )
                    
                    # Compute top1 accuracy on training
                    top1_train = get_n_accuracies_train(model, df, X_processed, y, 100)[0]
                    results_training.append({
                        'max_depth': d,
                        'n_estimators': n,
                        'learning_rate': lr,
                        'scale_pos_weight': spw,
                        'reg_lambda': reg,
                        'top1_accuracy': np.mean(top1_train)
                    })

                    # Compute top1 accuracy on validation
                    top1_val = get_n_accuracies_test(model, df, X_processed, y, 100)[0]
                    results_validation.append({
                        'max_depth': d,
                        'n_estimators': n,
                        'learning_rate': lr,
                        'scale_pos_weight': spw,
                        'reg_lambda': reg,
                        'top1_accuracy': np.mean(top1_val)
                    })

# Convert results to DataFrames
df_train_results = pd.DataFrame(results_training)
df_val_results = pd.DataFrame(results_validation)

# Display nicely sorted validation results
print("Validation Top-1 Accuracy (sorted):")
print(df_val_results.sort_values(by="top1_accuracy", ascending=False).head(10).to_string(index=False))

'''

results_training = []
results_validation = []
depths = [1, 2, 3, 4, 5,6]
lambdas = [0, 1, 5, 10]
for d in depths:
    for l2 in lambdas:
        model = XGBClassifier(
                        n_estimators=200,
                        max_depth=d,
                        learning_rate=0.01,
                        reg_lambda=l2,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=0,
                    )
        # Compute training accuracy
        top1_train = get_n_accuracies_train(model, df, X_processed, y, 100)[0]
        results_training.append({
            'max_depth': d,
            'lambda': l2,
            'top1_accuracy': np.mean(top1_train)
        })
        
        # Compute validation accuracy
        top1_val = get_n_accuracies_test(model, df, X_processed, y, 100)[0]
        results_validation.append({
            'max_depth': d,
            'lambda': l2,
            'top1_accuracy': np.mean(top1_val)
        })



df_train = pd.DataFrame(results_training)
df_val = pd.DataFrame(results_validation)

train_pivot = df_train.pivot(index='max_depth', columns='lambda', values='top1_accuracy')
val_pivot = df_val.pivot(index='max_depth', columns='lambda', values='top1_accuracy')

#Plot for validation 
plt.figure(figsize=(10, 6))
sns.heatmap(val_pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Top-1 Accuracy by max_depth and lambda values on validation set")
plt.xlabel("Regularization strength")
plt.ylabel("Max Depth")
plt.tight_layout()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'Top1_valid_XGB.png') 
plt.savefig(plot_path)
plt.close()

#Plot for training 
plt.figure(figsize=(10, 6))
sns.heatmap(train_pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Top-1 Accuracy by max_depth and lambda values on training set")
plt.xlabel("Regularization strength")
plt.ylabel("Max Depth")
plt.tight_layout()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'Top1_train_XGB.png') 
plt.savefig(plot_path)
plt.close()