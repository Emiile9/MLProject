import argparse
import os
import json
import pickle
from time import gmtime, strftime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluation import compute_topk_accuracy
from movies_preprocessing import full_processing
from sklearn.pipeline import Pipeline
from dataset_split import train_test_split_perso


parser = argparse.ArgumentParser(
        prog = "24/25 ML Project Emile Descroix",
        description = "Main Program for the 24/25 ML Project")

parser.add_argument("--dataset_path", type = str, default = "", help = "path to the dataset file")
parser.add_argument("--ml_method", type = str, default = "Logistic", help = "name of the ML method to use ('Logistic', 'Random Forest', 'Gradient Boosting')")
parser.add_argument("--l2_penalty", type = float, default = 1., help = "strength of the L2 penalty used when fitting the model")
parser.add_argument("--cv_nsplits", type = int, default = 5, help = "cross-validation: number of splits")
parser.add_argument("--save_dir", type = str, default = "", help = "where to save the model, the logs and the configuration")

args = parser.parse_args()

# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f)

# Loading the dataset
df = pd.read_csv(args.dataset_path)
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
y = df['winner']


# Build the model
if args.ml_method == "Logistic":
    model = LogisticRegression(C=args.l2_penalty, class_weight='balanced', max_iter=1000)
else:
    raise ValueError(f"Invalid value found for argument 'ml_method': found '{args.ml_method}'")


X_processed = full_processing(X, "median")
model.fit(X_processed, y)

# Save model
with open(path_model, 'wb') as f:
    pickle.dump({"model": model}, f)

'''
# Test model
lst_scores_mse = cross_val_score(pipeline, X, y, cv = args.cv_nsplits, scoring = "neg_mean_squared_error")
score_mse = sum(lst_scores_mse) / args.cv_nsplits

lst_scores_r2 = cross_val_score(pipeline, X, y, cv = args.cv_nsplits)
score_r2 = sum(lst_scores_r2) / args.cv_nsplits
'''

top1_acc, top3_acc = compute_topk_accuracy(model, X_processed, y, 3)
print(top1_acc, top3_acc)
# Store results
with open(path_logs, "w") as f:
    json.dump({"top1_acc": top1_acc,
        "top3_acc": top3_acc}, f)
