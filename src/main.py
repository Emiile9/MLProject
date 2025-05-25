import argparse
import os
import json
import pickle
from time import gmtime, strftime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from evaluation import compute_topk_accuracy
from movies_preprocessing import full_processing
from dataset_split import train_test_split_perso


parser = argparse.ArgumentParser(
        prog = "24/25 ML Project Emile Descroix",
        description = "Main Program for the 24/25 ML Project")

parser.add_argument("--dataset_path", type = str, default = "", help = "path to the dataset file")
parser.add_argument("--ml_method", type = str, default = "Logistic", help = "name of the ML method to use ('Logistic', 'Random Forest', 'Gradient Boosting')")
parser.add_argument("--l2_penalty", type = float, default = 1., help = "strength of the L2 penalty used when fitting the model")
parser.add_argument("--max_depth", type = float, default = 3, help = "max depth of trees estimators for Random Forest and XGBoost")
parser.add_argument("--n_estimators", type = float, default = 3, help = "number of trees estimators for Random Forest and XGBoost")
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
X_processed = full_processing(X, "median")
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split_perso(df, X_processed, y, 0.2)


# Build the model
if args.ml_method == "Logistic":
    model = LogisticRegression(C=args.l2_penalty, class_weight='balanced', max_iter=1000)
elif args.ml_method == "Random Forest":
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth= args.max_depth, class_weight='balanced')
elif args.ml_method == "XGBoost":
    model = model = XGBClassifier(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        learning_rate=0.01,
                        reg_lambda=args.l2_penalty,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=0,
                    )
else:
    raise ValueError(f"Invalid value found for argument 'ml_method': found '{args.ml_method}'")


X_processed = full_processing(X, "median")
model.fit(X_train, y)

# Save model
with open(path_model, 'wb') as f:
    pickle.dump({"model": model}, f)

#Test model for top1 and top3 accuracies
top1_acc, top3_acc = compute_topk_accuracy(model, X_processed, y, 3)

# Store results
with open(path_logs, "w") as f:
    json.dump({"top1_acc": top1_acc,
        "top3_acc": top3_acc}, f)
