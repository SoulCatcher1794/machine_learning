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

q2_rating_matrix = np.full((q1_n_users, q1_n_movies), np.nan)

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