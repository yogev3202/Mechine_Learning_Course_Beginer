# ---------------------------------------------------------------------------------
# Import Packages
# ---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------------------------------------------------------------------------------
# Set random seed for reproducibility
# ---------------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------------
# Generate Gaussian distributed data centered around different means
# ---------------------------------------------------------------------------------

cluster_1 = np.random.normal(loc=[2, 2], scale=0.7, size=(100, 2))
cluster_2 = np.random.normal(loc=[8, 3], scale=1.0, size=(100, 2))
cluster_3 = np.random.normal(loc=[5, 7], scale=0.6, size=(100, 2))
cluster_4 = np.random.normal(loc=[1, 8], scale=1.2, size=(100, 2))
cluster_5 = np.random.normal(loc=[7, 7], scale=0.8, size=(100, 2))

# Plot the Raw Data

plt.figure(figsize=(8, 6))
plt.scatter(cluster_1[:, 0], cluster_1[:, 1], label='Data Set 1', alpha=0.7)
plt.scatter(cluster_2[:, 0], cluster_2[:, 1], label='Data Set 2', alpha=0.7)
plt.scatter(cluster_3[:, 0], cluster_3[:, 1], label='Data Set 3', alpha=0.7)
plt.scatter(cluster_4[:, 0], cluster_4[:, 1], label='Data Set 4', alpha=0.7)
plt.scatter(cluster_5[:, 0], cluster_5[:, 1], label='Data Set 5', alpha=0.7)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original Gaussian Data')
plt.legend()
plt.grid(True)
# plt.show()


# ---------------------------------------------------------------------------------
# Find the Clusters from the Data
# ---------------------------------------------------------------------------------

# Combine the clusters to form the dataset
data = np.vstack((cluster_1, cluster_2, cluster_3, cluster_4, cluster_5))

# Number of clusters
k = 5

# Initialize and fit K-means model
kmeans = KMeans(n_clusters=k, random_state=1, init= 'random')
kmeans.fit(data)

# Get cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# K-means model Accuracy
# Inertia (WCSS)
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")
# Silhouette Score
silhouette_avg = silhouette_score(data, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

# Davies-Bouldin Index
dbi = davies_bouldin_score(data, kmeans.labels_)
print(f"Davies-Bouldin Index: {dbi}")

# Plotting the data points and centroids
plt.figure(figsize=(8, 6))
# Plot each cluster with a different color
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-means Clustering on Gaussian Data')

plt.text(0.02, 0.98, f'Inertia: {inertia:.2f}\nSilhouette Score: {silhouette_avg:.2f}\nDBI: {dbi:.2f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                                                        edgecolor="black", facecolor="white"))

plt.legend()
plt.grid(True)
plt.show()




