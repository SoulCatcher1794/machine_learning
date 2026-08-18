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
assert isinstance(q3_movie_means, np.ndarray), "q3_movie_means must be a numpy array."
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



# Grade Cell: Question 6
#
# Task: Implement the iterative matrix completion algorithm
#
# Instructions:
# 1. Start with mean-imputed matrix
# 2. Create a mask for observed entries (not NaN in original)
# 3. Iterate: SVD reconstruct, update only missing entries
# 4. Track convergence by monitoring change in imputed values

def q6_complete_matrix(matrix, rank, max_iter):
    # Fill missing values with column means (assume mean = 0 after centering)
    new_matrix = np.nan_to_num(matrix, nan=0.0)
    prev_matrix = new_matrix.copy()
    # Create a mask (index matrix) for observed entries (not NaN in original matrix)
    # Compliment of the NaN mask gives True for observed entries
    mask = ~np.isnan(matrix)
    # Tracking list of RMSE
    error_tracking = []

    for _ in range(max_iter):
        # Reconstruct the matrix using SVD
        reconstructed_matrix = q5_reconstruct_matrix(new_matrix, rank)
        # Update matrix with SVD reconstruction only for original missing entries
        new_matrix[~mask] = reconstructed_matrix[~mask]
        # Calculate RMSE on observed entries and append to error tracking list
        rmse = np.sqrt(np.mean( (np.mean(new_matrix) - np.mean(prev_matrix)) ** 2))
        error_tracking.append(round(float(rmse),4))
        # Update prev_matrix for the next iteration
        prev_matrix = new_matrix.copy()

    return new_matrix, error_tracking

q6_completed, q6_errors = q6_complete_matrix(q3_centered_matrix, 5, 20)
q6_final_error = q6_errors[-1]


# If all tests pass (there might be hidden tests), you will earn 15 points
# Test Cell: Question 6
assert callable(q6_complete_matrix), "q6_complete_matrix should be a callable function."
assert isinstance(q6_completed, np.ndarray), "q6_completed must be a numpy array."
assert (
    q6_completed.shape == q3_centered_matrix.shape
), "Completed matrix should have the same shape as input."
assert not np.isnan(
    q6_completed
).any(), "Completed matrix should not contain NaN values."
# Check that observed values are preserved
mask = ~np.isnan(q3_centered_matrix)
observed_diff = np.abs(q6_completed[mask] - q3_centered_matrix[mask])
assert observed_diff.max() < 0.01, (
    "Observed values should be preserved (not changed during completion). "
    "Make sure you only update missing entries."
)
assert isinstance(q6_final_error, float), "q6_final_error must be a float."
assert q6_final_error >= 0, "RMSE cannot be negative."
print(f"Completed matrix shape: {q6_completed.shape}")
print(f"Number of iterations: {len(q6_errors)}")
print(f"Final RMSE on observed: {q6_final_error}")



# Grade Cell: Question 7
#
# Task: Select optimal rank using validation by masking
#
# Instructions:
# 1. Create a validation mask: randomly hide 20% of observed entries
# 2. For each candidate rank, run completion and measure recovery on masked entries
# 3. Find the rank with lowest validation error

# Create a validation mask to create validation set of 20% of observed entries
# Random creates a distribution of values between 0 and 1, we compare it to 0.2 to get a boolean mask
validation_mask = np.random.rand(*q3_centered_matrix.shape) < 0.2
q7_validation_errors = {}
# Create list of candidate ranks to test
candidate_ranks = [1, 2, 3, 5, 10]

for rank in candidate_ranks:
    # Use Q4 mean-imputed matrix as the starting point for reconstruction
    masked_matrix = q4_mean_imputed.copy()
    # Use mask to hide observed entries and reconstruct matrix
    completed_matrix = q5_reconstruct_matrix(masked_matrix, rank)
    # Calculate rmse: complete vs reconstructed matrix
    rmse = np.sqrt(np.mean((completed_matrix[validation_mask] - q4_mean_imputed[validation_mask]) ** 2))
    q7_validation_errors[rank] = round(float(rmse), 4)

# Find the rank with the lowest validation error
q7_best_rank = min(q7_validation_errors, key=q7_validation_errors.get)
q7_best_error = q7_validation_errors[q7_best_rank]

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(
    q7_validation_errors, dict
), "q7_validation_errors must be a dictionary."
assert set(q7_validation_errors.keys()) == {
    1,
    2,
    3,
    5,
    10,
}, "Should test ranks [1, 2, 3, 5, 10]."
for rank, error in q7_validation_errors.items():
    assert isinstance(error, float), f"Error for rank {rank} must be a float."
    assert error > 0, f"Error for rank {rank} should be positive."
assert isinstance(q7_best_rank, int), "q7_best_rank must be an integer."
assert (
    q7_best_rank in candidate_ranks
), "q7_best_rank should be one of the tested ranks."
assert isinstance(q7_best_error, float), "q7_best_error must be a float."
assert (
    q7_best_error == q7_validation_errors[q7_best_rank]
), "q7_best_error should match the error for q7_best_rank."
print("Validation errors by rank:")
for rank in sorted(q7_validation_errors.keys()):
    marker = " <-- best" if rank == q7_best_rank else ""
    print(f"  Rank {rank}: {q7_validation_errors[rank]:.4f}{marker}")
print(f"\nBest rank: {q7_best_rank} with error: {q7_best_error:.4f}")



# Grade Cell: Question 8
#
# Task: Evaluate imputation quality against ground truth
#
# Instructions:
# 1. Load the ground truth matrix
# 2. Center it using the same movie means from Q3
# 3. Extract imputed and true values for originally missing entries
# 4. Compute correlation and RMSE

# Load the ground truth ratings from csv file
q8_ground_truth = pd.read_csv("ratings_ground_truth.csv").to_numpy()
# Center using the same movie means from Q3
q8_centered_ground_truth = q8_ground_truth - q3_movie_means
# Create a mask of the imputed values from original dataset
q8_missing_mask = np.isnan(q2_rating_matrix)
# Use mask to extract imputed and true values for originally missing entries
# Will use reconstructed matrix from Q5 to get imputed values
q8_imputed_missing = q6_completed[q8_missing_mask]
# Use mask to extract true values from complete dataset
q8_true_missing = q8_centered_ground_truth[q8_missing_mask]
# Calculate correlation between imputed and true values
q8_correlation = np.corrcoef(q8_imputed_missing, q8_true_missing).round(3)[0, 1]
# Calculate RMSE between imputed and true values
q8_rmse = np.sqrt(np.mean((q8_imputed_missing - q8_true_missing) ** 2)).round(3)


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
assert isinstance(q8_ground_truth, np.ndarray), "q8_ground_truth must be a numpy array."
assert (
    q8_ground_truth.shape == q2_rating_matrix.shape
), "Ground truth should have the same shape as the rating matrix."
assert isinstance(
    q8_imputed_missing, np.ndarray
), "q8_imputed_missing must be a numpy array."
assert isinstance(q8_true_missing, np.ndarray), "q8_true_missing must be a numpy array."
assert len(q8_imputed_missing) == len(
    q8_true_missing
), "Imputed and true arrays should have the same length."
assert (
    len(q8_imputed_missing) == q2_n_missing
), "Should have one imputed value per missing entry."
assert isinstance(q8_correlation, float), "q8_correlation must be a float."
assert -1 <= q8_correlation <= 1, "Correlation must be between -1 and 1."
assert isinstance(q8_rmse, float), "q8_rmse must be a float."
assert q8_rmse >= 0, "RMSE cannot be negative."
print(f"Number of missing entries evaluated: {len(q8_imputed_missing)}")
print(f"Correlation (imputed vs true): {q8_correlation}")
print(f"RMSE (imputed vs true): {q8_rmse}")



# Grade Cell: Question 9
#
# Task: Build a recommender system using the completed matrix
#
# Instructions:
# 1. Create a function that looks up the completed value and un-centers it
# 2. For user 0, predict ratings for all unrated movies
# 3. Sort by predicted rating to get top recommendations

def q9_predict_rating(user_id, movie_id, reconstructed_matrix, movie_means):
    # Look up the completed value for the user and movie
    completed_value = reconstructed_matrix[user_id, movie_id]
    # Un-center by adding back the movie mean
    predicted_rating = round(float(completed_value + movie_means[movie_id]), 3)
    return predicted_rating

# Use Q6 completed matrix for predictions
final_completed = q6_completed
q9_user_0_predictions = {}

# Get the list of unrated movies for user 0
unrated_movies = np.where(np.isnan(q2_rating_matrix[0]))[0]

# Predict ratings for all unrated movies for user 0
for movie_id in unrated_movies:
    predicted_rating = q9_predict_rating(0, movie_id, final_completed, q3_movie_means)
    q9_user_0_predictions[int(movie_id)] = predicted_rating

# Get the top 5 movie recommendations for user 0
q9_top_5_movies = sorted(q9_user_0_predictions, key=q9_user_0_predictions.get, reverse=True)[:5]



# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert callable(q9_predict_rating), "q9_predict_rating should be a callable function."
# Test the function
test_pred = q9_predict_rating(0, 0, final_completed, q3_movie_means)
assert isinstance(test_pred, float), "Prediction should be a float."
assert 1.0 <= test_pred <= 5.0, "Predicted rating should be in range [1, 5]."
assert isinstance(
    q9_user_0_predictions, dict
), "q9_user_0_predictions must be a dictionary."
assert len(q9_user_0_predictions) > 0, "User 0 should have some unrated movies."
for movie_id, rating in q9_user_0_predictions.items():
    assert isinstance(movie_id, int), "Movie IDs should be integers."
    assert isinstance(rating, float), "Ratings should be floats."
    assert 1.0 <= rating <= 5.0, "Ratings should be in [1, 5]."
assert isinstance(q9_top_5_movies, list), "q9_top_5_movies must be a list."
assert len(q9_top_5_movies) == 5, "Should recommend exactly 5 movies."
assert all(isinstance(m, int) for m in q9_top_5_movies), "Movie IDs should be integers."
# Top movies should be unrated
assert all(
    m in unrated_movies for m in q9_top_5_movies
), "Recommendations should be for unrated movies."
print(f"User 0 has {len(q9_user_0_predictions)} unrated movies")
print(f"Top 5 movie recommendations for User 0: {q9_top_5_movies}")
print("Predicted ratings for top 5:")
for movie_id in q9_top_5_movies:
    print(f"  Movie {movie_id}: {q9_user_0_predictions[movie_id]:.2f}")



# Grade Cell: Question 10
#
# Task: Create a complete matrix completion pipeline
#
# Instructions:
# 1. Combine centering, completion, and un-centering into one function
# 2. Return a dictionary with completed matrix, movie means, and metadata

def q10_matrix_completion_pipeline(rating_matrix, rank=5, max_iter=20):
    # Calculate movie means
    movie_means = np.nanmean(rating_matrix, axis=0)
    # Centering
    centered_matrix = rating_matrix - movie_means
    # SVD Reconstruction
    reconstructed_matrix, convergence_errors = q6_complete_matrix(centered_matrix, rank, max_iter)
    # Un-centering and fixing OOB values with clip function
    completed_matrix = np.clip(reconstructed_matrix + movie_means, 1.0, 5.0)
    # Count missing entries in the original matrix
    n_missing = int(np.isnan(rating_matrix).sum())
    
    return {
        "completed_matrix": completed_matrix.round(3),
        "movie_means": movie_means,
        "n_missing": n_missing,
        "rank": rank,
        "convergence_errors": convergence_errors,
    }

q10_pipeline_result = q10_matrix_completion_pipeline(q2_rating_matrix)
q10_result_keys = list(q10_pipeline_result.keys())

# If all tests pass (there might be hidden tests), you will earn 5 points
# Test Cell: Question 10
assert callable(
    q10_matrix_completion_pipeline
), "q10_matrix_completion_pipeline should be a callable function."
assert isinstance(q10_pipeline_result, dict), "Pipeline should return a dictionary."
expected_keys = {
    "completed_matrix",
    "movie_means",
    "n_missing",
    "rank",
    "convergence_errors",
}
assert (
    set(q10_result_keys) >= expected_keys
), f"Result should contain keys: {expected_keys}. Got: {set(q10_result_keys)}"
# Check completed matrix
completed = q10_pipeline_result["completed_matrix"]
assert (
    completed.shape == q2_rating_matrix.shape
), "Completed matrix should have same shape as input."
assert not np.isnan(completed).any(), "Completed matrix should not have any NaN values."
assert (completed >= 1.0).all() and (
    completed <= 5.0
).all(), "All ratings should be in range [1, 5]."
assert isinstance(q10_result_keys, list), "q10_result_keys must be a list."
print(f"Pipeline result keys: {q10_result_keys}")
print(f"Completed matrix shape: {completed.shape}")
print(f"Number of imputed entries: {q10_pipeline_result['n_missing']}")
print(f"Rank used: {q10_pipeline_result['rank']}")