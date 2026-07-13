import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("us_arrests.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "us_arrests.csv is missing from the lab directory. Please run the download "
        "script (w1_download_datasets.py) or ask the TA for assistance."
    )

# Grade Cell: Question 1
#
# Task: Load the USArrests dataset and explore its structure
#
# Instructions:
# 1. Read the CSV file using pd.read_csv() with index_col=0 to use state names as index
# 2. Store the shape as a tuple in q1_shape
# 3. Store the column names as a list in q1_columns

# your code here
df = pd.read_csv(_DATA_PATH, index_col=0)
q1_shape = df.shape
q1_columns = df.columns.to_list()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert isinstance(q1_shape, tuple), (
    "q1_shape must be a tuple. Use df.shape which returns (rows, cols)."
)
assert len(q1_shape) == 2, (
    "q1_shape should have 2 elements (rows, cols). "
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
print(f"Dataset shape: {q1_shape}")
print(f"Columns: {q1_columns}")



# Grade Cell: Question 2
#
# Task: Compute summary statistics for each feature
#
# Instructions:
# 1. Calculate the mean of each column
# 2. Calculate the standard deviation of each column (use ddof=0)
# 3. Round values to 2 decimal places
# 4. Store as dictionaries with column names as keys

# your code here
q2_means = {}
q2_stds = {}

for column in q1_columns:
    q2_means[column] = round(df[column].mean(), 2)
    q2_stds[column] = round(df[column].std(ddof=0), 2)



# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(q2_means, dict), (
    "q2_means must be a dictionary mapping feature names to mean values."
)
assert isinstance(q2_stds, dict), (
    "q2_stds must be a dictionary mapping feature names to std values."
)
assert len(q2_means) == 3, (
    "q2_means should have 3 entries, one for each feature."
)
assert len(q2_stds) == 3, (
    "q2_stds should have 3 entries, one for each feature."
)
assert all(isinstance(v, float) for v in q2_means.values()), (
    "All mean values should be floats. Use round() to ensure this."
)
assert all(v > 0 for v in q2_stds.values()), (
    "Standard deviations must be positive. Check your calculation."
)
print(f"Feature means: {q2_means}")
print(f"Feature stds: {q2_stds}")



# Grade Cell: Question 3
#
# Task: Compute the range (max - min) for each feature
#
# Instructions:
# 1. For each column, compute max() - min()
# 2. Round to 2 decimal places
# 3. Store as a dictionary

# your code here
q3_ranges = {}

for column in q1_columns:
    q3_ranges[column] = round((df[column].max() - df[column].min()), 2)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert isinstance(q3_ranges, dict), (
    "q3_ranges must be a dictionary mapping feature names to range values."
)
assert len(q3_ranges) == 3, (
    "q3_ranges should have 3 entries, one for each feature."
)
assert all(v > 0 for v in q3_ranges.values()), (
    "Range values must be positive. Check your max - min calculation."
)
# Check that Assault has the largest range (it should dominate distances)
max_range_feature = max(q3_ranges, key=q3_ranges.get)
print(f"Feature ranges: {q3_ranges}")
print(f"Feature with largest range: {max_range_feature}")



# Grade Cell: Question 4
#
# Task: Implement Euclidean distance function
#
# Instructions:
# 1. Implement the euclidean_distance function using the formula above
# 2. Use np.sqrt() and np.sum() for the calculation
# 3. Round the result to 3 decimal places
# 4. Compute distance between Alabama and Alaska

# your code here
def euclidean_distance(x, y):
      if len(x) != len(y):
           raise ValueError("Both variables must be vectors and of the same length")
      distance = np.sqrt(np.sum((x-y)**2))
      return round(float(distance), 3)

d_alabama = df.loc["Alabama"].to_numpy()
d_alaska = df.loc["Alaska"].to_numpy()     
q4_distance_al_ak = euclidean_distance(d_alabama, d_alaska)          

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert callable(euclidean_distance), (
    "euclidean_distance should be a function. Did you define it with 'def'?"
)
# Test with known values
test_x = np.array([1, 2, 3])
test_y = np.array([4, 5, 6])
test_dist = euclidean_distance(test_x, test_y)
assert isinstance(test_dist, float), (
    "euclidean_distance should return a float."
)
assert test_dist >= 0, (
    "Distance must be non-negative. Check your formula."
)
assert isinstance(q4_distance_al_ak, float), (
    "q4_distance_al_ak must be a float."
)
assert q4_distance_al_ak > 0, (
    "Distance between different states should be positive."
)
print(f"Test distance [1,2,3] to [4,5,6]: {test_dist}")
print(f"Distance Alabama to Alaska: {q4_distance_al_ak}")



# Grade Cell: Question 5
#
# Task: Implement Manhattan distance function
#
# Instructions:
# 1. Implement the manhattan_distance function using the formula above
# 2. Use np.sum() and np.abs() for the calculation
# 3. Round the result to 3 decimal places
# 4. Compute distance between Alabama and Alaska

# your code here
def manhattan_distance(x, y):
     if len(x) != len(y):
           raise ValueError("Both variables must be vectors and of the same length")
     distance = np.sum(np.abs(x-y))
     return round(float(distance), 3)

d_alabama = df.loc["Alabama"].to_numpy()
d_alaska = df.loc["Alaska"].to_numpy() 
q5_manhattan_al_ak = manhattan_distance(d_alabama, d_alaska)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert callable(manhattan_distance), (
    "manhattan_distance should be a function. Did you define it with 'def'?"
)
# Test with known values
test_x = np.array([1, 2, 3])
test_y = np.array([4, 5, 6])
test_manhattan = manhattan_distance(test_x, test_y)
assert isinstance(test_manhattan, float), (
    "manhattan_distance should return a float."
)
assert test_manhattan >= 0, (
    "Distance must be non-negative. Check your formula."
)
# Manhattan should be >= Euclidean for the same points
assert test_manhattan >= euclidean_distance(test_x, test_y), (
    "Manhattan distance should be >= Euclidean distance for the same points."
)
assert isinstance(q5_manhattan_al_ak, float), (
    "q5_manhattan_al_ak must be a float."
)
print(f"Test Manhattan distance [1,2,3] to [4,5,6]: {test_manhattan}")
print(f"Manhattan distance Alabama to Alaska: {q5_manhattan_al_ak}")



# Grade Cell: Question 6
#
# Task: Build a pairwise Euclidean distance matrix
#
# Instructions:
# 1. Create a 50x50 matrix of zeros
# 2. For each pair of states, compute Euclidean distance
# 3. Store distances in the matrix (it should be symmetric)

# your code here


# If all tests pass (there might be hidden tests), you will earn 15 points
# Test Cell: Question 6
assert isinstance(q6_dist_matrix, np.ndarray), (
    "q6_dist_matrix must be a numpy array."
)
assert q6_dist_matrix.shape == (50, 50), (
    f"Distance matrix should be 50x50, got {q6_dist_matrix.shape}. "
    "Make sure you're computing distances for all pairs of states."
)
# Check diagonal is zeros (distance to self)
assert np.allclose(np.diag(q6_dist_matrix), 0), (
    "Diagonal elements should be 0 (distance from a state to itself)."
)
# Check symmetry
assert np.allclose(q6_dist_matrix, q6_dist_matrix.T), (
    "Distance matrix should be symmetric. d(A,B) = d(B,A)."
)
# Check all values are non-negative
assert (q6_dist_matrix >= 0).all(), (
    "All distances must be non-negative."
)
print(f"Distance matrix shape: {q6_dist_matrix.shape}")
print(f"Diagonal (should be zeros): {np.diag(q6_dist_matrix)[:5]}")
print(f"Max distance: {q6_dist_matrix.max():.2f}")



# Grade Cell: Question 7
#
# Task: Standardize the data using StandardScaler
#
# Instructions:
# 1. Create a StandardScaler instance
# 2. Fit and transform the data
# 3. Convert back to DataFrame with original index and columns
# 4. Compute means and stds of the scaled data (should be ~0 and ~1)

# your code here


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(q7_scaled_data, pd.DataFrame), (
    "q7_scaled_data must be a pandas DataFrame."
)
assert q7_scaled_data.shape == df.shape, (
    "Scaled data should have same shape as original."
)
assert list(q7_scaled_data.columns) == list(df.columns), (
    "Scaled data should have same column names as original."
)
assert isinstance(q7_scaled_means, dict), (
    "q7_scaled_means must be a dictionary."
)
assert isinstance(q7_scaled_stds, dict), (
    "q7_scaled_stds must be a dictionary."
)
# Check that means are approximately 0
all_means_near_zero = all(abs(v) < 0.01 for v in q7_scaled_means.values())
assert all_means_near_zero, (
    "After standardization, all feature means should be ~0. "
    "Did you use StandardScaler correctly?"
)
# Check that stds are approximately 1
all_stds_near_one = all(abs(v - 1.0) < 0.01 for v in q7_scaled_stds.values())
assert all_stds_near_one, (
    "After standardization, all feature stds should be ~1. "
    "Did you use StandardScaler correctly?"
)
print(f"Scaled means (should be ~0): {q7_scaled_means}")
print(f"Scaled stds (should be ~1): {q7_scaled_stds}")



# Grade Cell: Question 8
#
# Task: Build distance matrix on standardized data
#
# Instructions:
# 1. Compute pairwise Euclidean distances on q7_scaled_data
# 2. Extract the distance between Alabama and Alaska

# your code here


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
assert isinstance(q8_scaled_dist_matrix, np.ndarray), (
    "q8_scaled_dist_matrix must be a numpy array."
)
assert q8_scaled_dist_matrix.shape == (50, 50), (
    f"Distance matrix should be 50x50, got {q8_scaled_dist_matrix.shape}."
)
assert np.allclose(np.diag(q8_scaled_dist_matrix), 0), (
    "Diagonal elements should be 0."
)
assert np.allclose(q8_scaled_dist_matrix, q8_scaled_dist_matrix.T), (
    "Distance matrix should be symmetric."
)
assert isinstance(q8_al_ak_scaled, float), (
    "q8_al_ak_scaled must be a float."
)
print(f"Unstandardized Alabama-Alaska distance: {q4_distance_al_ak}")
print(f"Standardized Alabama-Alaska distance: {q8_al_ak_scaled}")
print(f"Ratio (unstandardized / standardized): {q4_distance_al_ak / q8_al_ak_scaled:.2f}")



# Grade Cell: Question 9
#
# Task: Find the two most similar states based on standardized distances
#
# Instructions:
# 1. Copy the distance matrix and set diagonal to infinity
# 2. Find the indices of the minimum value
# 3. Get the state names corresponding to those indices

# your code here


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert isinstance(q9_most_similar_pair, tuple), (
    "q9_most_similar_pair must be a tuple of two state names."
)
assert len(q9_most_similar_pair) == 2, (
    "q9_most_similar_pair should contain exactly 2 state names."
)
assert all(isinstance(s, str) for s in q9_most_similar_pair), (
    "Both elements in q9_most_similar_pair should be strings (state names)."
)
assert q9_most_similar_pair[0] != q9_most_similar_pair[1], (
    "The two states should be different. Did you exclude the diagonal?"
)
assert isinstance(q9_min_distance, float), (
    "q9_min_distance must be a float."
)
assert q9_min_distance > 0, (
    "Minimum distance should be positive (not self-distance)."
)
print(f"Most similar states: {q9_most_similar_pair}")
print(f"Distance between them: {q9_min_distance}")



# Grade Cell: Question 10
#
# Task: Implement k-nearest neighbors lookup
#
# Instructions:
# 1. Get the row of distances for the given state
# 2. Sort indices by distance (excluding the state itself)
# 3. Return the top k state names

# your code here

# If all tests pass (there might be hidden tests), you will earn 5 points
# Test Cell: Question 10
assert callable(find_k_nearest), (
    "find_k_nearest should be a function."
)
assert isinstance(q10_ca_neighbors, list), (
    "q10_ca_neighbors must be a list of state names."
)
assert len(q10_ca_neighbors) == 3, (
    "q10_ca_neighbors should contain exactly 3 neighbors."
)
assert all(isinstance(s, str) for s in q10_ca_neighbors), (
    "All neighbors should be strings (state names)."
)
assert "California" not in q10_ca_neighbors, (
    "California should not be in its own neighbor list."
)
# Test the function with another state
test_neighbors = find_k_nearest("Texas", 2)
assert len(test_neighbors) == 2, (
    "find_k_nearest(Texas, 2) should return 2 neighbors."
)
assert "Texas" not in test_neighbors, (
    "A state should not appear in its own neighbor list."
)
print(f"California's 3 nearest neighbors: {q10_ca_neighbors}")
print(f"Texas's 2 nearest neighbors: {test_neighbors}")