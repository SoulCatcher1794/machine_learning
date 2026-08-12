import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("iris.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "iris.csv is missing from the lab directory. Please run the download "
        "script (w3_download_datasets.py) or ask the TA for assistance."
    )

# Grade Cell: Question 1
#
# Task: Load the Iris dataset and explore its structure
#
# Instructions:
# 1. Read the CSV file using pd.read_csv()
# 2. Store the shape as a tuple in q1_shape
# 3. Store the column names as a list in q1_columns
# 4. Store the number of samples (rows) as an integer in q1_n_samples

df = pd.read_csv(_DATA_PATH)
q1_shape = df.shape
q1_columns = df.columns.tolist()
q1_n_samples = df.shape[0]

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert isinstance(
    q1_shape, tuple
), "q1_shape must be a tuple. Use df.shape which returns (rows, cols)."
assert len(q1_shape) == 2, (
    "q1_shape should have 2 elements (n_samples, n_features). "
    "Make sure you're using .shape, not .shape[0]."
)
assert (
    q1_shape[0] > 0 and q1_shape[1] > 0
), "Shape values must be positive. Is your CSV loading correctly?"
assert isinstance(
    q1_columns, list
), "q1_columns must be a list. Use df.columns.tolist() to convert."
assert (
    len(q1_columns) == q1_shape[1]
), "Number of columns in q1_columns should match q1_shape[1]."
assert isinstance(q1_n_samples, int), "q1_n_samples must be an integer."
print(f"Dataset shape: {q1_shape}")
print(f"Number of samples: {q1_n_samples}")
print(f"Features: {q1_columns}")



# Grade Cell: Question 2
#
# Task: Standardize the data using StandardScaler
#
# Instructions:
# 1. Create a StandardScaler instance
# 2. Fit and transform the data (df.values or df as input)
# 3. Compute the column means and stds of the scaled data to verify

sc = StandardScaler()
q2_scaled_data = sc.fit_transform(df)
# Compute the means and stds of each feature/column (axis=0)
q2_scaled_means = q2_scaled_data.mean(axis=0).round(3)
q2_scaled_stds = q2_scaled_data.std(axis=0, ddof=0).round(3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(q2_scaled_data, np.ndarray), "q2_scaled_data must be a numpy array."
assert (
    q2_scaled_data.shape == q1_shape
), f"Scaled data should have same shape as original: {q1_shape}."
assert isinstance(q2_scaled_means, np.ndarray), "q2_scaled_means must be a numpy array."
assert isinstance(q2_scaled_stds, np.ndarray), "q2_scaled_stds must be a numpy array."
# Check that all means are approximately 0
assert np.allclose(q2_scaled_means, 0, atol=0.01), (
    "After standardization, all feature means should be ~0. "
    "Did you use StandardScaler correctly?"
)
# Check that all stds are approximately 1
assert np.allclose(q2_scaled_stds, 1, atol=0.01), (
    "After standardization, all feature stds should be ~1. "
    "Did you use StandardScaler correctly?"
)
print(f"Scaled data shape: {q2_scaled_data.shape}")
print(f"Scaled means (should be ~0): {q2_scaled_means}")
print(f"Scaled stds (should be ~1): {q2_scaled_stds}")



# Grade Cell: Question 3
#
# Task: Fit a K-Means model with K=3 clusters
#
# Instructions:
# 1. Create a KMeans instance with n_clusters=3, random_state=RANDOM_STATE, n_init=10
# 2. Fit the model to q2_scaled_data
# 3. Get the cluster labels from .labels_
# 4. Get the inertia from .inertia_
# 5. Count samples per cluster

Kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
q3_kmeans = Kmeans.fit(q2_scaled_data)
q3_labels = Kmeans.labels_
q3_inertia = round(Kmeans.inertia_, 2)
q3_cluster_counts = {}

for label in np.unique(q3_labels):
    # Count the instaces that match each label and assign that value to each key/label
    q3_cluster_counts[label] = np.sum(q3_labels == label)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert hasattr(
    q3_kmeans, "cluster_centers_"
), "KMeans model doesn't appear to be fitted. Did you call kmeans.fit()?"
assert isinstance(q3_labels, np.ndarray), "q3_labels must be a numpy array."
assert (
    len(q3_labels) == q1_n_samples
), f"q3_labels should have {q1_n_samples} entries, one per sample."
assert set(q3_labels) == {0, 1, 2}, (
    "With K=3, labels should be 0, 1, and 2. " "Check that n_clusters=3."
)
assert isinstance(q3_inertia, float), "q3_inertia must be a float."
assert q3_inertia > 0, "Inertia (within-cluster sum of squares) must be positive."
assert isinstance(
    q3_cluster_counts, dict
), "q3_cluster_counts must be a dictionary mapping label to count."
assert (
    sum(q3_cluster_counts.values()) == q1_n_samples
), "Total samples across clusters should equal n_samples."
print(f"Cluster labels: {q3_labels[:10]}... (first 10)")
print(f"Inertia: {q3_inertia}")
print(f"Samples per cluster: {q3_cluster_counts}")



# Grade Cell: Question 4
#
# Task: Implement the elbow method to find optimal K
#
# Instructions:
# 1. Loop through K = 1 to 10
# 2. Fit KMeans with each K and record inertia
# 3. Use random_state=RANDOM_STATE and n_init=10 for consistency
# 4. Identify the elbow point (where inertia stops decreasing rapidly)

def calculateInertia(k):
    Kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    Kmeans.fit(q2_scaled_data)
    inertia = round(Kmeans.inertia_, 2)
    return inertia

ks = range(1,11)
# Use list comprenhension to calculate inertia for K=1 to K=10
q4_inertias = [calculateInertia(k) for k in range(1, 11)]
plt.figure()
plt.plot(range(1, 11), q4_inertias)
# Taken from visual inspection of the plot
q4_elbow_k = 3


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert isinstance(q4_inertias, list), "q4_inertias must be a list of floats."
assert len(q4_inertias) == 10, "q4_inertias should have 10 values (for K=1 to K=10)."
# Inertia should be decreasing
assert all(q4_inertias[i] >= q4_inertias[i + 1] for i in range(len(q4_inertias) - 1)), (
    "Inertia should decrease as K increases. "
    "More clusters = less within-cluster variance."
)
assert isinstance(q4_elbow_k, int), "q4_elbow_k must be an integer."
assert 2 <= q4_elbow_k <= 5, (
    "The elbow for iris data should be between K=2 and K=5. "
    "Look for where the slope changes most dramatically."
)
print(f"Inertias for K=1 to K=10: {q4_inertias}")
print(f"Elbow K: {q4_elbow_k}")

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), q4_inertias, "bo-", linewidth=2, markersize=8)
plt.axvline(x=q4_elbow_k, color="red", linestyle="--", label=f"Elbow at K={q4_elbow_k}")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# Grade Cell: Question 5
#
# Task: Compute silhouette scores for different values of K
#
# Instructions:
# 1. Loop through K = 2 to 6 (silhouette requires at least 2 clusters)
# 2. Fit KMeans and get cluster labels
# 3. Compute silhouette_score for each K
# 4. Find the K with the highest score

q5_silhouette_scores = {}

for k in range(2, 7):
    Kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    # Fit the model and predict cluster labels in a single step
    labels = Kmeans.fit_predict(q2_scaled_data)
    score = silhouette_score(q2_scaled_data, labels)
    q5_silhouette_scores[k] = round(score, 3)

q5_best_k_silhouette = max(q5_silhouette_scores, key=q5_silhouette_scores.get)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert isinstance(
    q5_silhouette_scores, dict
), "q5_silhouette_scores must be a dictionary mapping K to score."
assert (
    len(q5_silhouette_scores) == 5
), "q5_silhouette_scores should have 5 entries (K=2 to K=6)."
assert all(
    2 <= k <= 6 for k in q5_silhouette_scores.keys()
), "Keys should be K values from 2 to 6."
assert all(
    -1 <= score <= 1 for score in q5_silhouette_scores.values()
), "Silhouette scores must be between -1 and 1."
assert isinstance(q5_best_k_silhouette, int), "q5_best_k_silhouette must be an integer."
assert (
    q5_best_k_silhouette in q5_silhouette_scores
), "q5_best_k_silhouette must be one of the K values tested."
print(f"Silhouette scores: {q5_silhouette_scores}")
print(f"Best K by silhouette: {q5_best_k_silhouette}")



# Grade Cell: Question 6
#
# Task: Choose the optimal K and justify your decision
#
# Instructions:
# 1. Consider evidence from both elbow method and silhouette analysis
# 2. Choose a K value and explain why
# 3. Compute final inertia and silhouette for your chosen K

q6_chosen_k = q4_elbow_k  # Based on elbow method
q6_justification = "Chose K based on the elbow method, where inertia stops decreasing rapidly."

q6_kmeans = KMeans(n_clusters=q6_chosen_k, random_state=RANDOM_STATE, n_init=10)
q6_labels = q6_kmeans.fit_predict(q2_scaled_data)
q6_final_inertia = round(q6_kmeans.inertia_, 2)
q6_final_silhouette = round(silhouette_score(q2_scaled_data, q6_labels), 3)
    
# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 6
assert isinstance(q6_chosen_k, int), "q6_chosen_k must be an integer."
assert (
    2 <= q6_chosen_k <= 5
), "Your chosen K should be reasonable (between 2 and 5 for iris data)."
assert isinstance(
    q6_justification, str
), "q6_justification must be a string explaining your choice."
assert (
    len(q6_justification) >= 20
), "Please provide a more detailed justification (at least 20 characters)."
assert isinstance(q6_final_inertia, float), "q6_final_inertia must be a float."
assert q6_final_inertia > 0, "Inertia must be positive."
assert isinstance(q6_final_silhouette, float), "q6_final_silhouette must be a float."
assert -1 <= q6_final_silhouette <= 1, "Silhouette score must be between -1 and 1."
print(f"Chosen K: {q6_chosen_k}")
print(f"Justification: {q6_justification}")
print(f"Final inertia: {q6_final_inertia}")
print(f"Final silhouette: {q6_final_silhouette}")



# Grade Cell: Question 7
#
# Task: Profile clusters by computing feature means
#
# Instructions:
# 1. Fit KMeans with your chosen K (q6_chosen_k)
# 2. Add cluster labels to a copy of the original data
# 3. Compute mean of each feature per cluster
# 4. Identify the largest cluster and the dominant feature for cluster 0

q7_kmeans = KMeans(n_clusters=q6_chosen_k, random_state=RANDOM_STATE, n_init=10)
q7_labels = q7_kmeans.fit_predict(q2_scaled_data)
# Create a DataFrame with the scaled data
q7_data = pd.DataFrame(q2_scaled_data.copy(), columns=q1_columns)
# Assign the cluster labels to a new column in the DataFrame
q7_data["cluster"] = q7_labels
# Compute the mean of each feature of each cluster using the groupby method
q7_cluster_means = q7_data.groupby("cluster")[q1_columns].mean().round(3)
# Get the size of each cluster and find the cluster with the maximum size
# idxmax() returns the index of the first occurrence of the maximum value
q7_largest_cluster = int(q7_data.groupby("cluster").size().idxmax())
# Identify the dominant feature for cluster 0 by finding the feature with the maximum mean value
# Use loc to access the row corresponding to cluster 0 and idxmax() to find the column name of the maximum value
q7_cluster_0_dominant_feature = q7_cluster_means.loc[0].idxmax()


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(
    q7_cluster_means, pd.DataFrame
), "q7_cluster_means must be a pandas DataFrame."
assert (
    q7_cluster_means.shape[0] == q6_chosen_k
), f"q7_cluster_means should have {q6_chosen_k} rows (one per cluster)."
assert q7_cluster_means.shape[1] == len(
    q1_columns
), f"q7_cluster_means should have {len(q1_columns)} columns (one per feature)."
assert isinstance(
    q7_largest_cluster, int
), "q7_largest_cluster must be an integer (cluster label)."
assert (
    0 <= q7_largest_cluster < q6_chosen_k
), f"q7_largest_cluster should be between 0 and {q6_chosen_k - 1}."
assert isinstance(
    q7_cluster_0_dominant_feature, str
), "q7_cluster_0_dominant_feature must be a string (feature name)."
assert (
    q7_cluster_0_dominant_feature in q1_columns
), "q7_cluster_0_dominant_feature must be one of the original feature names."
print("Cluster means (standardized scale):")
print(q7_cluster_means.round(2))
print(f"\nLargest cluster: {q7_largest_cluster}")
print(f"Cluster 0 dominant feature: {q7_cluster_0_dominant_feature}")



# Grade Cell: Question 8
#
# Task: Visualize clusters using PCA projection
#
# Instructions:
# 1. Fit PCA with n_components=2 to the standardized data
# 2. Transform the data to 2D
# 3. Create a scatter plot colored by cluster labels
# 4. Compute total variance explained by the 2 PCs

pca = PCA(n_components=2, random_state=RANDOM_STATE)
q8_pca = pca.fit(q2_scaled_data)
# Get the transformed data projected onto the first 2 principal components
q8_pca_data = pca.transform(q2_scaled_data)
# Compute the total variance explained by all PCs (in this case 2)
q8_variance_explained = float(pca.explained_variance_ratio_.sum().round(3))

plt.figure()
# Create a scatter plot of the PCA-transformed data, colored by cluster labels
# Column 1 is PC1, Column 2 is PC2 used as x and y axes respectively
plt.scatter(q8_pca_data[:, 0], q8_pca_data[:, 1], c=q7_labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
q8_plot_created = True

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
assert hasattr(
    q8_pca, "components_"
), "q8_pca doesn't appear to be fitted. Did you call pca.fit() or pca.fit_transform()?"
assert isinstance(q8_pca_data, np.ndarray), "q8_pca_data must be a numpy array."
assert q8_pca_data.shape == (
    q1_n_samples,
    2,
), f"q8_pca_data should have shape ({q1_n_samples}, 2)."
assert isinstance(
    q8_variance_explained, float
), "q8_variance_explained must be a float."
assert 0 < q8_variance_explained <= 1, "Variance explained should be between 0 and 1."
assert q8_variance_explained > 0.5, (
    "2 PCs should explain at least 50% of variance for iris data. "
    "Check your PCA implementation."
)
assert (
    q8_plot_created is True
), "Set q8_plot_created to True after creating the scatter plot."
print(f"PCA data shape: {q8_pca_data.shape}")
print(f"Variance explained by 2 PCs: {q8_variance_explained*100:.1f}%")



# Grade Cell: Question 9
#
# Task: Analyze K-Means stability across different initializations
#
# Instructions:
# 1. Run KMeans with K=3, n_init=1 for 10 different random_state values (0-9)
# 2. Record the inertia from each run
# 3. Compute mean and std of inertias
# 4. Determine if results are stable (low coefficient of variation)

q9_kmeans = KMeans(n_clusters=3, n_init=1)
q9_inertias = []

for random_state in range(10):
    q9_kmeans.set_params(random_state=random_state)
    q9_kmeans.fit(q2_scaled_data)
    inertia = round(q9_kmeans.inertia_, 2)
    q9_inertias.append(inertia)

q9_mean_inertia = float(np.mean(q9_inertias).round(3))
q9_std_inertia = float(np.std(q9_inertias).round(3))
q9_is_stable = bool((q9_std_inertia / q9_mean_inertia) < 0.05)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert isinstance(q9_inertias, list), "q9_inertias must be a list of floats."
assert (
    len(q9_inertias) == 10
), "q9_inertias should have 10 values (one per random state)."
assert all(i > 0 for i in q9_inertias), "All inertia values must be positive."
assert isinstance(q9_mean_inertia, float), "q9_mean_inertia must be a float."
assert isinstance(q9_std_inertia, float), "q9_std_inertia must be a float."
assert q9_std_inertia >= 0, "Standard deviation cannot be negative."
assert isinstance(q9_is_stable, bool), "q9_is_stable must be a boolean."
print(f"Inertias from 10 runs: {q9_inertias}")
print(f"Mean inertia: {q9_mean_inertia}")
print(f"Std inertia: {q9_std_inertia}")
print(f"Coefficient of variation: {q9_std_inertia/q9_mean_inertia*100:.2f}%")
print(f"Is stable (CV < 5%): {q9_is_stable}")



# Grade Cell: Question 10
#
# Task: Implement a reusable clustering analysis function
#
# Instructions:
# 1. Create a function that takes data and K as inputs
# 2. Fit KMeans and compute labels, inertia, and silhouette score
# 3. Return results as a dictionary
# 4. Test with K=3 and K=4

def q10_cluster_analysis(data, k):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels= kmeans.fit_predict(data)
    inertia = round(kmeans.inertia_, 3)
    silhouette = round(silhouette_score(data, labels), 3)
    # Create a DataFrame to simplify cluster size calculation
    df = pd.DataFrame(data, columns=q1_columns)
    df['cluster'] = labels
    cluster_sizes = df.groupby('cluster').size().to_list()

    return {
        "k": k,
        "labels": labels.tolist(),
        "inertia": inertia,
        "silhouette": silhouette,
        "cluster_sizes": cluster_sizes,
    }

q10_result_k3 = q10_cluster_analysis(q2_scaled_data, k=3)
q10_result_k4 = q10_cluster_analysis(q2_scaled_data, k=4)
q10_better_k = 3 if q10_result_k3["silhouette"] > q10_result_k4["silhouette"] else 4

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 10
assert callable(q10_cluster_analysis), "q10_cluster_analysis should be a function."

# Test function output structure
test_result = q10_cluster_analysis(q2_scaled_data, k=2)
assert isinstance(test_result, dict), "Function should return a dictionary."
assert "k" in test_result, "Result should contain 'k'."
assert "labels" in test_result, "Result should contain 'labels'."
assert "inertia" in test_result, "Result should contain 'inertia'."
assert "silhouette" in test_result, "Result should contain 'silhouette'."
assert "cluster_sizes" in test_result, "Result should contain 'cluster_sizes'."

# Test K=3 result
assert isinstance(q10_result_k3, dict), "q10_result_k3 must be a dictionary."
assert q10_result_k3["k"] == 3, "q10_result_k3 should have k=3."
assert len(q10_result_k3["labels"]) == q1_n_samples, "Labels should match sample count."

# Test K=4 result
assert isinstance(q10_result_k4, dict), "q10_result_k4 must be a dictionary."
assert q10_result_k4["k"] == 4, "q10_result_k4 should have k=4."

# Test better_k
assert isinstance(q10_better_k, int), "q10_better_k must be an integer."
assert q10_better_k in [3, 4], "q10_better_k should be 3 or 4."

print("K=3 results:")
print(f"  Inertia: {q10_result_k3['inertia']}")
print(f"  Silhouette: {q10_result_k3['silhouette']}")
print(f"  Cluster sizes: {q10_result_k3['cluster_sizes']}")
print("\nK=4 results:")
print(f"  Inertia: {q10_result_k4['inertia']}")
print(f"  Silhouette: {q10_result_k4['silhouette']}")
print(f"  Cluster sizes: {q10_result_k4['cluster_sizes']}")
print(f"\nBetter K by silhouette: {q10_better_k}")