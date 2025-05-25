🎬 Oscar Best Picture Prediction

This project aims to predict the Best Picture Oscar winner using various machine learning techniques. It evaluates logistic regression, random forest, and gradient boosting models on historical Oscar nominee data. The goal is not only to classify the winner correctly but also to rank nominees by their likelihood of winning, using metrics like Top-1 and Top-3 accuracy.

🧠 Models Supported

Logistic Regression \\
Random Forest \\
Gradient Boosting (XGBoost) \\
📁 Dataset

The dataset should contain features such as:

IMDb / Rotten Tomatoes / Letterboxd scores
Budget
Country, Language
Genre
Previous Oscar-winning cast or crew
and more...
Each year must have exactly one winner.

🚀 Running the Code

python main.py --dataset_path path/to/your/data.csv \
               --ml_method RandomForest \
               --l2_penalty 0.5 \
               --max_depth 5 \
               --n_estimators 100 \
               --cv_nsplits 5 \
               --save_dir outputs/
🧾 Command-Line Arguments

Argument	Type	Default	Description
--dataset_path	str	""	Path to the CSV file containing the dataset
--ml_method	str	"Logistic"	Which ML method to use. Choose among: 'Logistic', 'RandomForest', or 'GradientBoosting'
--l2_penalty	float	1.0	L2 regularization strength used for logistic regression or as reg_lambda for XGBoost
--max_depth	float	3	Maximum depth of trees for Random Forest and Gradient Boosting
--n_estimators	float	3	Number of trees in Random Forest or Gradient Boosting
--cv_nsplits	int	5	Number of cross-validation folds
--save_dir	str	""	Directory to save model artifacts, logs, and configuration files
📊 Evaluation Metrics

Top-1 Accuracy: Percentage of years where the top-predicted nominee matches the actual winner
Top-3 Accuracy: Percentage of years where the actual winner is within the top 3 predicted nominees

📦 Output

When training is complete, results are saved in the directory specified by --save_dir, including:

Trained model
Evaluation metrics