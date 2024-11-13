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
n_points = 300
# Add random noise
noise = 0.1  # Adjust noise level as desired
x_noisy =  np.random.normal(0, noise, n_points)
y_noisy =  np.random.normal(0, noise, n_points)


angles = np.random.uniform(0, 2 * np.pi, n_points)

# Calculate x and y coordinates for random points on the circle
center_x1 = 0 ; center_y1 = 0 ; radius1 = 3
x1_random = center_x1 + radius1 * np.cos(angles) + x_noisy
y1_random = center_y1 + radius1 * np.sin(angles) + y_noisy
cluster_1 = np.array([x1_random,y1_random]).transpose()

center_x2 = 2 ; center_y2 = 2 ; radius2 = 1
x2_random = center_x2 + radius2 * np.cos(angles) + x_noisy
y2_random = center_y2 + radius2 * np.sin(angles) + y_noisy
cluster_2 = np.array([x2_random,y2_random]).transpose()



# Plot the Raw Data

plt.figure(figsize=(8, 6))
plt.scatter(cluster_1[:, 0], cluster_1[:, 1], label='Data Set 1', alpha=0.7)
plt.scatter(cluster_2[:, 0], cluster_2[:, 1], label='Data Set 1', alpha=0.7)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original Gaussian Data')
plt.legend()
plt.grid(True)
# plt.ion()
# plt.show()



# ---------------------------------------------------------------------------------
# Find the Clusters from the Data
# ---------------------------------------------------------------------------------

# Combine the clusters to form the dataset
data = np.vstack((cluster_1, cluster_2))

# Number of clusters
k = 2

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




