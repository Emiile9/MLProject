import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from movies_preprocessing import full_processing
from sklearn.metrics import silhouette_score
import os

plot_folder = os.path.join('..', 'plots')
df = pd.read_csv('../data/final_oscar_data.csv')
features = ["year_film", "genres", "averageRating", "dir_won_before", "budget", "nb_actor_won_before","won_bafta","won_gg_drama","won_gg_comedy", "runtimeMinutes"]
X = df[features]
X_processed = full_processing(X, "median")
X_processed = X_processed.drop(columns=['year_film'])
y = df['winner']

#Find best k
sil_score_k = []
kmax = 10
ks = [k for k in range(2,kmax+1)]
for k in ks:
    kmeans = KMeans(n_clusters=k).fit(X_processed)
    labels = kmeans.labels_
    sil_score_k.append(silhouette_score(X_processed, labels, metric = "euclidean"))

plt.plot(ks, sil_score_k)
plt.xlabel('Number of clusters')
plt.ylabel('Silouhette Score')
plot_path = os.path.join(plot_folder, 'k_choice.png') 
plt.savefig(plot_path)
plt.close()

#PCA with 2 componenents

kmeans = KMeans(n_clusters=2).fit(X_processed)
cluster_labels = kmeans.labels_
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.75)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering with KMeans (k=2) projected with PCA")
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'clusters_PCA_plot.png') 
plt.savefig(plot_path)
plt.close()


winners_mask = y == 1
nominees_mask = y == 0

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(X_pca[nominees_mask, 0], X_pca[nominees_mask, 1], 
            c='red', label='Nominees', alpha=0.5)

plt.scatter(X_pca[winners_mask, 0], X_pca[winners_mask, 1], 
            c='green', label='Winners', alpha=0.7)

plt.title("PCA of Movies (Winners vs Nominees)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(False)
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'winners_PCA_plot.png') 
plt.savefig(plot_path)
plt.close()