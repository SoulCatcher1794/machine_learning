import pathlib
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("ratings.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "ratings.csv is missing from the lab directory. Please run the download "
        "script (w5_download_datasets.py) or ask the TA for assistance."
    )

# Grade Cell: Question 1
#
# Task: Load the ratings dataset and explore its structure
#
# Instructions:
# 1. Read the CSV file using pd.read_csv()
# 2. Count total ratings, unique users, and unique movies
# 3. Find the minimum and maximum rating values

q1_df = pd.read_csv(_DATA_PATH)
q1_n_ratings = q1_df.shape[0]
q1_n_users = q1_df["user_id"].unique().size
q1_n_movies = q1_df["movie_id"].unique().size
q1_rating_range = (q1_df["rating"].min(), q1_df["rating"].max())

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert isinstance(
    q1_df, pd.DataFrame
), "q1_df must be a pandas DataFrame. Use pd.read_csv() to load the data."
assert set(q1_df.columns) == {"user_id", "movie_id", "rating"}, (
    "DataFrame should have columns: user_id, movie_id, rating. "
    "Check that you loaded the correct file."
)
assert isinstance(q1_n_ratings, int), "q1_n_ratings must be an integer."
assert q1_n_ratings > 0, "There should be at least some ratings."
assert isinstance(q1_n_users, int), "q1_n_users must be an integer."
assert isinstance(q1_n_movies, int), "q1_n_movies must be an integer."
assert isinstance(q1_rating_range, tuple), "q1_rating_range must be a tuple."
assert len(q1_rating_range) == 2, "q1_rating_range should be (min, max)."
assert q1_rating_range[0] <= q1_rating_range[1], "Min should be <= max."
print(f"Total ratings: {q1_n_ratings}")
print(f"Unique users: {q1_n_users}")
print(f"Unique movies: {q1_n_movies}")
print(f"Rating range: {q1_rating_range}")



# Grade Cell: Question 2
#
# Task: Create the rating matrix and compute sparsity
#
# Instructions:
# 1. Initialize a matrix of NaN values with shape (n_users, n_movies)
# 2. Fill in the observed ratings from the DataFrame
# 3. Count NaN values and compute sparsity as n_missing / total_entries

# Fill the matrix with NaN values
q2_rating_matrix = np.full((q1_n_users, q1_n_movies), np.nan)

# Iterate over each row and fill the matrix with the existing data
for review in q1_df.itertuples(index=False):
    i = review.user_id
    j = review.movie_id
    q2_rating_matrix[i, j] = review.rating

# isnan function provides an array with True for each value that is NaN
# The sum function will sum all "True" values in the array and provide total
q2_n_missing = int(np.isnan(q2_rating_matrix).sum())
q2_sparsity = round(q2_n_missing/q2_rating_matrix.size, 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(
    q2_rating_matrix, np.ndarray
), "q2_rating_matrix must be a numpy array."
assert q2_rating_matrix.shape == (q1_n_users, q1_n_movies), (
    f"Matrix shape should be ({q1_n_users}, {q1_n_movies}). "
    "Rows are users, columns are movies."
)
assert np.isnan(q2_rating_matrix).any(), (
    "Matrix should contain NaN values for missing ratings. "
    "Initialize with np.full(..., np.nan)."
)
assert isinstance(q2_n_missing, int), "q2_n_missing must be an integer."
assert q2_n_missing > 0, "There should be missing entries."
assert isinstance(q2_sparsity, float), "q2_sparsity must be a float."
assert (
    0 < q2_sparsity < 1
), "Sparsity should be between 0 and 1 (fraction of missing entries)."
# The sum of observed ratings should match our original count
n_observed = (~np.isnan(q2_rating_matrix)).sum()
assert n_observed == q1_n_ratings, (
    "Number of observed entries should match q1_n_ratings. "
    "Check your matrix filling logic."
)
print(f"Rating matrix shape: {q2_rating_matrix.shape}")
print(f"Missing entries: {q2_n_missing}")
print(f"Sparsity: {q2_sparsity:.1%}")



# Grade Cell: Question 3
#
# Task: Center the rating matrix by movie means
#
# Instructions:
# 1. Compute the mean of each column, ignoring NaN values (use np.nanmean)
# 2. Subtract these means from the observed values
# 3. Verify that the centered matrix has mean ~0

# Calculate the mean of each movie (column)
q3_movie_means = np.nanmean(q2_rating_matrix, axis=0)
# Center the matrix by subtracting the movie means from each observed rating
q3_centered_matrix = q2_rating_matrix - q3_movie_means
# Calculate the overall mean of the centered matrix
q3_centered_mean = np.nanmean(q3_centered_matrix)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert isinstance(++, np.ndarray), "q3_movie_means must be a numpy array."
assert q3_movie_means.shape == (
    q1_n_movies,
), f"q3_movie_means should have length {q1_n_movies} (one mean per movie)."
assert not np.isnan(
    q3_movie_means
).any(), (
    "Movie means should not contain NaN. Each movie should have at least one rating."
)
assert isinstance(
    q3_centered_matrix, np.ndarray
), "q3_centered_matrix must be a numpy array."
assert (
    q3_centered_matrix.shape == q2_rating_matrix.shape
), "Centered matrix should have the same shape as the original."
# Should still have NaN in the same places
assert (
    np.isnan(q3_centered_matrix).sum() == q2_n_missing
), "Centering should not change which entries are missing."
assert isinstance(q3_centered_mean, float), "q3_centered_mean must be a float."
assert abs(q3_centered_mean) < 0.1, (
    f"Centered mean is {q3_centered_mean}, expected ~0. "
    "Did you subtract movie means from all observed values?"
)
print(f"Movie means (first 5): {q3_movie_means[:5].round(2)}")
print(f"Centered mean: {q3_centered_mean}")



# Grade Cell: Question 4
#
# Task: Perform mean imputation on the centered matrix
#
# Instructions:
# 1. Replace all NaN values with 0 (since data is centered)
# 2. Verify no NaN values remain
# 3. Compute the range of values in the imputed matrix

# Replace NaN values with 0 in the centered matrix
q4_mean_imputed = np.nan_to_num(q3_centered_matrix, nan=0.0)
# Check if any NaN values remain, will return True if any NaN values are present
q4_has_nan = bool(np.isnan(q4_mean_imputed).any())
q4_imputed_range = (q4_mean_imputed.min().round(3), q4_mean_imputed.max().round(3))

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert isinstance(q4_mean_imputed, np.ndarray), "q4_mean_imputed must be a numpy array."
assert (
    q4_mean_imputed.shape == q3_centered_matrix.shape
), "Imputed matrix should have the same shape."
assert isinstance(q4_has_nan, bool), "q4_has_nan must be a boolean."
assert q4_has_nan is False, (
    "After imputation, there should be no NaN values. "
    "Use np.nan_to_num() or np.where() to replace NaN."
)
assert isinstance(q4_imputed_range, tuple), "q4_imputed_range must be a tuple."
assert len(q4_imputed_range) == 2, "q4_imputed_range should be (min, max)."
# The imputed values (which were 0) should bring the mean closer to 0
imputed_mean = q4_mean_imputed.mean()
assert abs(imputed_mean) < 0.5, "Mean of imputed matrix should be close to 0."
print(f"Imputed matrix shape: {q4_mean_imputed.shape}")
print(f"Contains NaN: {q4_has_nan}")
print(f"Value range: {q4_imputed_range}")



# Grade Cell: Question 5
#
# Task: Implement SVD-based matrix reconstruction
#
# Instructions:
# 1. Use np.linalg.svd to decompose the matrix
# 2. Keep only the top M singular values and corresponding vectors
# 3. Reconstruct using U_M @ diag(S_M) @ V_M^T
# 4. Compute RMSE as sqrt(mean((original - reconstructed)^2))

def q5_reconstruct_matrix(matrix, rank):
    # Get the SVD decomposition of the matrix into the components U, S, V
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    # Reconstruct the matrix truncating to the specified rank
    # @ is the matrix multiplication operator in Python
    reconstructed_matrix = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]
    return reconstructed_matrix

q5_reconstructed = q5_reconstruct_matrix(q4_mean_imputed, rank=5)
q5_reconstruction_error = np.round(float(np.sqrt(np.mean((q4_mean_imputed - q5_reconstructed)**2))), 4)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert callable(
    q5_reconstruct_matrix
), "q5_reconstruct_matrix should be a callable function."
# Test the function with a simple matrix
test_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
test_result = q5_reconstruct_matrix(test_matrix, rank=1)
assert (
    test_result.shape == test_matrix.shape
), "Function should return a matrix of the same shape."
assert isinstance(
    q5_reconstructed, np.ndarray
), "q5_reconstructed must be a numpy array."
assert (
    q5_reconstructed.shape == q4_mean_imputed.shape
), "Reconstructed matrix should have the same shape as input."
assert not np.isnan(
    q5_reconstructed
).any(), "Reconstructed matrix should not contain NaN."
assert isinstance(
    q5_reconstruction_error, float
), "q5_reconstruction_error must be a float."
assert q5_reconstruction_error >= 0, "RMSE cannot be negative."
assert q5_reconstruction_error < 1, (
    "Reconstruction error seems too large. "
    "A rank-5 approximation should capture most of the variance."
)
print(f"Reconstructed matrix shape: {q5_reconstructed.shape}")
print(f"Reconstruction RMSE: {q5_reconstruction_error}")