import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("wine.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "wine.csv is missing from the lab directory. Please run the download "
        "script (w2_download_datasets.py) or ask the TA for assistance."
    )

# Grade Cell: Question 1
#
# Task: Load the Wine dataset and explore its structure
#
# Instructions:
# 1. Read the CSV file using pd.read_csv()
# 2. Store the shape as a tuple in q1_shape
# 3. Store the column names as a list in q1_columns
# 4. Store the number of features (columns) as an integer in q1_n_features

df = pd.read_csv(_DATA_PATH)
q1_shape = df.shape
q1_columns = df.columns.to_list()
q1_n_features = len(q1_columns)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert isinstance(q1_shape, tuple), (
    "q1_shape must be a tuple. Use df.shape which returns (rows, cols)."
)
assert len(q1_shape) == 2, (
    "q1_shape should have 2 elements (n_samples, n_features). "
    "Make sure you're using .shape, not .shape[0]."
)
assert q1_shape[0] > 0 and q1_shape[1] > 0, (
    "Shape values must be positive. Is your CSV loading correctly?"
)
assert isinstance(q1_columns, list), (
    "q1_columns must be a list. Use df.columns.tolist() to convert."
)
assert len(q1_columns) == q1_shape[1], (
    "Number of columns in q1_columns should match q1_shape[1]."
)
assert isinstance(q1_n_features, int), (
    "q1_n_features must be an integer."
)
print(f"Dataset shape: {q1_shape}")
print(f"Number of features: {q1_n_features}")
print(f"Features: {q1_columns}")


# Grade Cell: Question 2
#
# Task: Compute summary statistics for each feature
#
# Instructions:
# 1. Calculate the mean of each column and round to 2 decimals
# 2. Calculate the standard deviation of each column (use ddof=0) and round to 2 decimals
# 3. Calculate the range (max - min) for each column
# 4. Find the feature with the largest range

q2_means = df[q1_columns].mean().round(2).to_dict()
q2_stds = df[q1_columns].std(ddof=0).round(2).to_dict()
q2_ranges = ( df[q1_columns].max() - df[q1_columns].min() ).round(2).to_dict()
# Use max function using the dictionary values (from get function) as key
q2_max_range_feature = max(q2_ranges, key=q2_ranges.get)


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(q2_means, dict), (
    "q2_means must be a dictionary mapping feature names to mean values."
)
assert isinstance(q2_stds, dict), (
    "q2_stds must be a dictionary mapping feature names to std values."
)
assert len(q2_means) == q1_n_features, (
    f"q2_means should have {q1_n_features} entries, one for each feature."
)
assert len(q2_stds) == q1_n_features, (
    f"q2_stds should have {q1_n_features} entries, one for each feature."
)
assert all(isinstance(v, float) for v in q2_means.values()), (
    "All mean values should be floats. Use round() to ensure this."
)
assert all(v > 0 for v in q2_stds.values()), (
    "Standard deviations must be positive. Check your calculation."
)
assert isinstance(q2_max_range_feature, str), (
    "q2_max_range_feature must be a string (the feature name)."
)
assert q2_max_range_feature in q1_columns, (
    "q2_max_range_feature must be one of the feature names."
)
print(f"Feature means: {q2_means}")
print(f"Feature stds: {q2_stds}")
print(f"Feature with largest range: {q2_max_range_feature}")



# Grade Cell: Question 3
#
# Task: Standardize the data using StandardScaler
#
# Instructions:
# 1. Create a StandardScaler instance
# 2. Fit and transform the data (df.values or df as input)
# 3. Compute the column means and stds of the scaled data to verify

sc = StandardScaler()
q3_scaled_data = sc.fit_transform(df)
scaled_data = pd.DataFrame(q3_scaled_data, columns=df.columns, index=df.index)
q3_scaled_means = scaled_data[q1_columns].mean().round(2).to_numpy()
q3_scaled_stds = scaled_data[q1_columns].std(ddof=0).round(2).to_numpy()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert isinstance(q3_scaled_data, np.ndarray), (
    "q3_scaled_data must be a numpy array."
)
assert q3_scaled_data.shape == q1_shape, (
    f"Scaled data should have same shape as original: {q1_shape}."
)
assert isinstance(q3_scaled_means, np.ndarray), (
    "q3_scaled_means must be a numpy array."
)
assert isinstance(q3_scaled_stds, np.ndarray), (
    "q3_scaled_stds must be a numpy array."
)
# Check that all means are approximately 0
assert np.allclose(q3_scaled_means, 0, atol=0.01), (
    "After standardization, all feature means should be ~0. "
    "Did you use StandardScaler correctly?"
)
# Check that all stds are approximately 1
assert np.allclose(q3_scaled_stds, 1, atol=0.01), (
    "After standardization, all feature stds should be ~1. "
    "Did you use StandardScaler correctly?"
)
print(f"Scaled data shape: {q3_scaled_data.shape}")
print(f"Scaled means (should be ~0): {q3_scaled_means}")
print(f"Scaled stds (should be ~1): {q3_scaled_stds}")

# Grade Cell: Question 4
#
# Task: Fit a PCA model to the standardized data
#
# Instructions:
# 1. Create a PCA instance (without specifying n_components to keep all)
# 2. Fit the PCA model to q3_scaled_data
# 3. Access the explained_variance_ratio_ attribute

pca = PCA()
q4_pca = pca.fit(q3_scaled_data)
q4_n_components = pca.n_components_
q4_explained_variance_ratio = pca.explained_variance_ratio_


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert hasattr(q4_pca, "components_"), (
    "PCA model doesn't appear to be fitted. Did you call pca.fit()?"
)
assert isinstance(q4_n_components, int), (
    "q4_n_components must be an integer."
)
assert q4_n_components == q1_n_features, (
    f"With all components kept, n_components should equal n_features ({q1_n_features}). "
    "Did you fit PCA without specifying n_components?"
)
assert isinstance(q4_explained_variance_ratio, np.ndarray), (
    "q4_explained_variance_ratio must be a numpy array."
)
assert len(q4_explained_variance_ratio) == q4_n_components, (
    "explained_variance_ratio should have one entry per component."
)
assert np.isclose(q4_explained_variance_ratio.sum(), 1.0, atol=0.001), (
    "Explained variance ratios should sum to 1.0 (100% of variance). "
    "Check that you're using the standardized data."
)
print(f"Number of components: {q4_n_components}")
print(f"Variance explained by each component:")
for i, var in enumerate(q4_explained_variance_ratio):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")



# Grade Cell: Question 5
#
# Task: Compute individual and cumulative variance explained
#
# Instructions:
# 1. Use q4_explained_variance_ratio to get individual PVE
# 2. Use np.cumsum() to compute cumulative PVE
# 3. Round values to 3 decimal places
# 4. Compute the sum of PC1 and PC2 variance

q5_pve = q4_explained_variance_ratio.round(3).tolist()
q5_cumulative_pve = np.cumsum(q4_explained_variance_ratio).round(3).tolist()
q5_pc1_pc2_total = float(q5_pve[0] + q5_pve[1])


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert isinstance(q5_pve, list), (
    "q5_pve must be a list of floats."
)
assert isinstance(q5_cumulative_pve, list), (
    "q5_cumulative_pve must be a list of floats."
)
assert len(q5_pve) == q4_n_components, (
    f"q5_pve should have {q4_n_components} entries."
)
assert len(q5_cumulative_pve) == q4_n_components, (
    f"q5_cumulative_pve should have {q4_n_components} entries."
)
# PVE values should be decreasing
assert all(q5_pve[i] >= q5_pve[i+1] for i in range(len(q5_pve)-1)), (
    "PVE values should be in decreasing order (PC1 explains most, then PC2, etc.)."
)
# Cumulative should be increasing
assert all(q5_cumulative_pve[i] <= q5_cumulative_pve[i+1] for i in range(len(q5_cumulative_pve)-1)), (
    "Cumulative PVE should be increasing."
)
# Last cumulative value should be ~1.0
assert abs(q5_cumulative_pve[-1] - 1.0) < 0.01, (
    "Final cumulative PVE should be ~1.0 (all variance explained)."
)
assert isinstance(q5_pc1_pc2_total, float), (
    "q5_pc1_pc2_total must be a float."
)
print(f"Individual PVE: {q5_pve}")
print(f"Cumulative PVE: {q5_cumulative_pve}")
print(f"PC1 + PC2 explain {q5_pc1_pc2_total*100:.1f}% of total variance")

# Grade Cell: Question 6
#
# Task: Find minimum components to explain 80% of variance
#
# Instructions:
# 1. Use the cumulative variance from Q5
# 2. Find the first index where cumulative variance >= 0.8
# 3. Remember: the number of components is index + 1

q6_threshold = 0.8
q6_n_components_80 = 0

for index, value in enumerate(q5_cumulative_pve):
    if(value >= q6_threshold):
        q6_n_components_80 = index + 1
        break
    
q6_actual_variance = q5_cumulative_pve[q6_n_components_80 - 1]


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 6
assert q6_threshold == 0.8, (
    "q6_threshold should be 0.8 (80% variance threshold)."
)
assert isinstance(q6_n_components_80, int), (
    "q6_n_components_80 must be an integer."
)
assert 1 <= q6_n_components_80 <= q4_n_components, (
    f"Number of components should be between 1 and {q4_n_components}."
)
assert isinstance(q6_actual_variance, float), (
    "q6_actual_variance must be a float."
)
assert q6_actual_variance >= q6_threshold, (
    f"Actual variance ({q6_actual_variance}) should be >= threshold ({q6_threshold}). "
    "Did you find the right number of components?"
)
# Check that one fewer component doesn't meet threshold
if q6_n_components_80 > 1:
    prev_variance = sum(q5_pve[:q6_n_components_80-1])
    assert prev_variance < q6_threshold, (
        "You might have selected more components than necessary. "
        "Find the MINIMUM number needed to reach 80%."
    )
print(f"Threshold: {q6_threshold*100:.0f}%")
print(f"Components needed: {q6_n_components_80}")
print(f"Actual variance explained: {q6_actual_variance*100:.1f}%")



# Grade Cell: Question 7
#
# Task: Examine and interpret PCA loadings
#
# Instructions:
# 1. Access the loadings from pca.components_
# 2. For PC1 (index 0), find the feature with largest |loading|
# 3. For PC2 (index 1), find the feature with largest |loading|
# 4. Use q1_columns to get feature names

# Loadings are the weights of the original features per each principal component (shape: PCs x features)
# Rows are the principal components, columns are the original features
q7_loadings = q4_pca.components_

# Find the index of the feature with the largest absolute loading for PC1 and PC2
q7_pc1_top_index = np.argmax(np.abs(q7_loadings[0]))
q7_pc2_top_index = np.argmax(np.abs(q7_loadings[1]))

# Get the feature labels of the features that contribute most to PC1 and PC2
q7_pc1_top_feature = q1_columns[q7_pc1_top_index]
q7_pc2_top_feature = q1_columns[q7_pc2_top_index]

# Get the loading value for the top feature of PC1
q7_pc1_top_loading = float(q7_loadings[0, q7_pc1_top_index])

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(q7_loadings, np.ndarray), (
    "q7_loadings must be a numpy array."
)
assert q7_loadings.shape == (q4_n_components, q1_n_features), (
    f"Loadings shape should be ({q4_n_components}, {q1_n_features})."
)
# Loadings should be unit vectors (rows have norm 1)
for i in range(min(3, q4_n_components)):
    row_norm = np.linalg.norm(q7_loadings[i, :])
    assert np.isclose(row_norm, 1.0, atol=0.01), (
        f"Row {i} of loadings should have norm 1.0 (unit vector), got {row_norm:.3f}."
    )
assert isinstance(q7_pc1_top_feature, str), (
    "q7_pc1_top_feature must be a string (feature name)."
)
assert q7_pc1_top_feature in q1_columns, (
    "q7_pc1_top_feature must be one of the original feature names."
)
assert isinstance(q7_pc2_top_feature, str), (
    "q7_pc2_top_feature must be a string (feature name)."
)
assert q7_pc2_top_feature in q1_columns, (
    "q7_pc2_top_feature must be one of the original feature names."
)
assert isinstance(q7_pc1_top_loading, float), (
    "q7_pc1_top_loading must be a float."
)
assert -1 <= q7_pc1_top_loading <= 1, (
    "Loadings should be between -1 and 1."
)
print(f"Loadings shape: {q7_loadings.shape}")
print(f"PC1 top feature: {q7_pc1_top_feature} (loading: {q7_pc1_top_loading})")
print(f"PC2 top feature: {q7_pc2_top_feature}")



# Grade Cell: Question 8
#
# Task: Transform data to principal component space
#
# Instructions:
# 1. Use pca.transform() on the standardized data
# 2. Create a DataFrame with column names "PC1", "PC2", etc.
# 3. Find the range (min, max) of PC1 scores

q8_scores = pca.transform(q3_scaled_data)
# Use list comprehension to create column names for the DataFrame
q8_columns = [f"PC{i}" for i in range(1, q8_scores.shape[1] + 1)]
q8_scores_df = pd.DataFrame(q8_scores, columns=q8_columns)
q8_pc1_range = (
    float(q8_scores_df["PC1"].min().round(2)),
    float(q8_scores_df["PC1"].max().round(2))
)


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
assert isinstance(q8_scores, np.ndarray), (
    "q8_scores must be a numpy array."
)
assert q8_scores.shape == (q1_shape[0], q4_n_components), (
    f"Scores shape should be ({q1_shape[0]}, {q4_n_components})."
)
assert isinstance(q8_scores_df, pd.DataFrame), (
    "q8_scores_df must be a pandas DataFrame."
)
assert q8_scores_df.shape == q8_scores.shape, (
    "DataFrame should have same shape as scores array."
)
assert "PC1" in q8_scores_df.columns, (
    "DataFrame columns should be named 'PC1', 'PC2', etc."
)
assert "PC2" in q8_scores_df.columns, (
    "DataFrame columns should be named 'PC1', 'PC2', etc."
)
assert isinstance(q8_pc1_range, tuple), (
    "q8_pc1_range must be a tuple (min, max)."
)
assert len(q8_pc1_range) == 2, (
    "q8_pc1_range should have exactly 2 elements (min, max)."
)
assert q8_pc1_range[0] < q8_pc1_range[1], (
    "First element should be min, second should be max."
)
print(f"Scores shape: {q8_scores.shape}")
print(f"DataFrame columns: {list(q8_scores_df.columns[:5])}...")
print(f"PC1 range: {q8_pc1_range}")



# Grade Cell: Question 9
#
# Task: Visualize PCA results and identify outliers
#
# Instructions:
# 1. Create a scatter plot of PC1 vs PC2
# 2. Calculate correlation between PC1 and PC2 scores
# 3. Find the sample furthest from origin (largest sqrt(PC1^2 + PC2^2))

# Create scatter plot of PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(q8_scores_df["PC1"], q8_scores_df["PC2"], alpha=0.7)
plt.title("PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
q9_plot_created = True

# Use the correlation coefficient function to compute correlation between PC1 and PC2
# Get the value of the correlation matrix which can be either [0, 1] or [1, 0]
q9_correlation = np.corrcoef(q8_scores_df["PC1"], q8_scores_df["PC2"])[0, 1]
q9_pc1_pc2_corr = round(float(q9_correlation), 3)

# Calculate the point furthest from the origin using Euclidean distance
q9_furthest_distance = np.sqrt(q8_scores_df["PC1"]**2 + q8_scores_df["PC2"]**2)
# Find the index of the maximum distance
q9_furthest_from_origin = int(np.argmax(q9_furthest_distance))

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert q9_plot_created is True, (
    "Set q9_plot_created to True after creating the scatter plot."
)
assert isinstance(q9_pc1_pc2_corr, float), (
    "q9_pc1_pc2_corr must be a float."
)
assert -0.1 <= q9_pc1_pc2_corr <= 0.1, (
    f"PC1 and PC2 should be uncorrelated (r ~ 0), got {q9_pc1_pc2_corr}. "
    "Principal components are orthogonal by construction."
)
assert isinstance(q9_furthest_from_origin, int), (
    "q9_furthest_from_origin must be an integer (sample index)."
)
assert 0 <= q9_furthest_from_origin < q1_shape[0], (
    f"Index should be between 0 and {q1_shape[0]-1}."
)
print(f"Plot created: {q9_plot_created}")
print(f"Correlation between PC1 and PC2: {q9_pc1_pc2_corr}")
print(f"Sample furthest from origin: index {q9_furthest_from_origin}")



# Grade Cell: Question 10
#
# Task: Implement PCA reconstruction and compute reconstruction error
#
# Instructions:
# 1. Create a function that fits PCA with k components and computes reconstruction MSE
# 2. MSE = mean((original - reconstructed)^2)
# 3. Compute MSE for all components, 2 components, and 5 components

# Define function to calculate reconstruction error for given number of PCs
def q10_reconstruction_error(k):
    pca = PCA(n_components=k)
    transformed_data = pca.fit_transform(q3_scaled_data)
    reconstructed_data = pca.inverse_transform(transformed_data)
    mse = np.mean((q3_scaled_data - reconstructed_data) ** 2).round(4)
    return mse

q10_mse_all_components = q10_reconstruction_error(q1_n_features)
q10_mse_2_components = q10_reconstruction_error(2)
q10_mse_5_components = q10_reconstruction_error(5)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 10
assert callable(q10_reconstruction_error), (
    "q10_reconstruction_error should be a function."
)
# Test the function
test_mse = q10_reconstruction_error(3)
assert isinstance(test_mse, float), (
    "Function should return a float."
)
assert test_mse >= 0, (
    "MSE must be non-negative."
)

assert isinstance(q10_mse_all_components, float), (
    "q10_mse_all_components must be a float."
)
assert q10_mse_all_components < 0.0001, (
    "MSE with all components should be ~0 (perfect reconstruction). "
    f"Got {q10_mse_all_components}."
)
assert isinstance(q10_mse_2_components, float), (
    "q10_mse_2_components must be a float."
)
assert isinstance(q10_mse_5_components, float), (
    "q10_mse_5_components must be a float."
)
# More components should give lower MSE
assert q10_mse_2_components > q10_mse_5_components, (
    "MSE with 2 components should be higher than with 5 components. "
    "More components = better reconstruction."
)
assert q10_mse_5_components > q10_mse_all_components, (
    "MSE with 5 components should be higher than with all components."
)
print(f"Reconstruction MSE with all components: {q10_mse_all_components}")
print(f"Reconstruction MSE with 2 components: {q10_mse_2_components}")
print(f"Reconstruction MSE with 5 components: {q10_mse_5_components}")