import matplotlib.pyplot as plt
import polars as pl 
import seaborn as sns 
import os

df = pl.read_csv('../data/final_data.csv')
winners_df = df.filter(pl.col("winner") == 1)
nominees_df = df.filter(pl.col("winner") == 0)

#Checking Ratings Distribution 
overall_ratings = df['averageRating']
winners_ratings = winners_df['averageRating']
nominees_ratings = nominees_df['averageRating']

#distribution plot
plt.figure(figsize=(8, 5))
sns.kdeplot(nominees_ratings, color='red', label='Nominees Distribution', fill=False)
sns.kdeplot(winners_ratings, color='green', label='Winners Distribution', fill=False)
plt.title('Rating Distribution for Nominees and Winners', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plot_folder = os.path.join('..', 'plots')
plot_path = os.path.join(plot_folder, 'ratings_plot.png') 
plt.savefig(plot_path)
plt.close()

#Same for budget 
overall_budget = df['budget']
winners_budget = winners_df['budget']
nominees_budget = nominees_df['budget']

#distribution plot
plt.figure(figsize=(8, 5))
sns.kdeplot(nominees_budget, color='red', label='Nominees Distribution', fill=False)
sns.kdeplot(winners_budget, color='green', label='Winners Distribution', fill=False)
plt.title('Budget Distribution for Nominees and Winners', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plot_path = os.path.join(plot_folder, 'budget_plot.png') 
plt.savefig(plot_path)
plt.close()
print('Hello')