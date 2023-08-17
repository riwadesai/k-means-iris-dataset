import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the Iris dataset (Replace this with the actual Iris dataset)
# The dataset should include 4 numeric variables and 1 categorical variable (SpeciesType)
# Example: data = pd.read_csv('iris.csv')

# Sample Iris dataset (replace this with your actual data)
data = pd.read_csv("/Users/riwadesai/Documents/mathfordatascience/Iris_Data.txt")

# Select the numeric features for k-means clustering
numeric_features = data.iloc[:, :-1].values

# Select the species column
species = data.iloc[:, -1].values

# Step 1: Apply k-means cluster analysis for k = 2, 3, 4 and report cluster statistics

# Function to perform k-means clustering and return cluster statistics
def kmeans_clustering(k, data):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    cluster_ids = kmeans.labels_
    centroids = kmeans.cluster_centers_
    within_cluster_variances = kmeans.inertia_
    return cluster_ids, centroids, within_cluster_variances

# Apply k-means clustering for k = 2, 3, 4
k_values = [2, 3, 4]
for k in k_values:
    cluster_ids, centroids, within_cluster_variances = kmeans_clustering(k, numeric_features)
    print(f"K = {k}")
    print("Cluster IDs:", cluster_ids)
    print("Centroids:", centroids)
    print("Within Cluster Variances:", within_cluster_variances)
    print()

# Step 2: Prepare a scree plot by plotting SSE(k) against k (number of clusters) for k = 2, 3, 4.

# Function to calculate the within-cluster sum of squares (SSE) for different k values
def calculate_sse(data, k_values):
    sse = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    return sse

# Calculate SSE values for k = 2, 3, 4
sse_values = calculate_sse(numeric_features, k_values)

# Plot the scree plot
plt.plot(k_values, sse_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within Cluster Sum of Squares (SSE)')
plt.title('Scree Plot for K-Means')
plt.show()

# Function to generate a frequency table of species type vs. cluster ID for k = 3
def generate_frequency_table(cluster_ids, species):
    species_names = np.unique(species)
    frequency_table = np.zeros((len(species_names), 3), dtype=int)

    for cluster_id in range(3):
        cluster_species = species[cluster_ids == cluster_id]
        for species_name in species_names:
            frequency_table[species_name == species_names, cluster_id] = np.sum(cluster_species == species_name)

    return frequency_table

# Perform k-means clustering for k = 3
k = 3
cluster_ids, _, _ = kmeans_clustering(k, numeric_features)

# Generate the frequency table
frequency_table = generate_frequency_table(cluster_ids, species)
print("Frequency Table:")
print(frequency_table)
# Function to perform hierarchical clustering with average distance method and generate 3 clusters
def hierarchical_clustering(data, k):
    linkage_matrix = linkage(data, method='average')
    cluster_ids = fcluster(linkage_matrix, k, criterion='maxclust')
    return cluster_ids

# Perform hierarchical clustering with k = 3
hierarchical_cluster_ids = hierarchical_clustering(numeric_features, k)

# Compare the two clustering solutions
# we'll simply print the cluster assignments for comparison.
print("K-Means Cluster IDs:", cluster_ids)
print("Hierarchical Cluster IDs:", hierarchical_cluster_ids)


# Step 1: Standardize the numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Step 2: Encode the categorical variable "SpeciesType"
label_encoder = LabelEncoder()
encoded_species = label_encoder.fit_transform(species)

# Step 3: Combine the scaled numeric features and encoded species data
combined_data = np.column_stack((scaled_data, encoded_species))

# Step 4: Perform k-means clustering with k = 3 using the combined data
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_ids = kmeans.fit_predict(combined_data)

# Print the cluster IDs for each data point
print("Cluster IDs with species type:")
print(cluster_ids)