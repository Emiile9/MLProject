import matplotlib.pyplot as plt
import polars as pl 
import seaborn as sns 

df = pl.read_csv('../data/final_data.csv')
winners_df = df.filter(pl.col("winner") == 1)
#Checking Ratings Distribution 
overall_ratings = df['averageRating']
winners_ratings = winners_df['averageRating']
#general plot 
plt.figure(figsize=(10, 6))
plt.hist(overall_ratings, bins=20, edgecolor='black', color='skyblue')

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Display the plot
plt.show()

#winners plot 
plt.figure(figsize=(10, 6))
plt.hist(winners_ratings, bins=20, edgecolor='black', color='skyblue')

plt.title('Winners Ratings')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Display the plot
plt.show()
print('Hello')