import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

RANDOM_STATE: int = 42  # global seed for full reproducibility
np.random.seed(RANDOM_STATE)

# Lab setup
_DATA_PATH = pathlib.Path("california_housing.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "california_housing.csv is missing from the lab directory. Please download it or ask the TA "
        "for assistance."
    )

# Part 1: Data loading and exploration
def load_housing() -> pd.DataFrame:
    """Load the California Housing data from ``california_housing.csv``.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the 8 predictors plus the target column ``MedHouseVal``.
    """
    df = pd.read_csv("california_housing.csv")
    return df

# Compute the answer required by the autograder
q1_shape: Tuple[int, int] = load_housing().shape

# Part 2: Data analysis
df = load_housing()
q1_limit = df["MedInc"].quantile(0.25)
q3_limit = df["MedInc"].quantile(0.75)
top = df[df["MedInc"] >= q3_limit] # All data that is at the top 25%
bottom = df[df["MedInc"] <= q1_limit] # All data that is at the bottom 25%

difference = top["MedHouseVal"].mean() - bottom["MedHouseVal"].mean()
q2_income_value_gap = round(float(difference), 3)

# Part 3: Model training and evaluation
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
n_train = len(X_train)
n_test = len(X_test)
q3_split_counts = (n_train, n_test)

# Part 4: KNN regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_prediction_knn = knn.predict(X_test)

mse_knn = mean_squared_error(y_test, y_prediction_knn)
q4_knn5_rmse = round(float(np.sqrt(mse_knn)), 3)

def knn_rmse(k: int) -> float: # helper function to compute RMSE for any K
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)

    mse = mean_squared_error(y_test, y_prediction)
    return round(float(np.sqrt(mse)), 3)

# Part 5: Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_prediction_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_prediction_lr)
q6_linreg_rmse = round(float(np.sqrt(mse_lr)), 3)

# Part 6: Cross-validation
def cross_val_knn(k_values: List[int]) -> Dict[int, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results: Dict[int, float] = {}

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
        mean_rmse = float(-cv_scores.mean())
        cv_results[k] = round(mean_rmse, 3)

    return cv_results

# Part 7: Choose best K and evaluate on test set
cv_rmse_by_k = cross_val_knn([1, 3, 5, 7, 9])
q8_best_k = min(cv_rmse_by_k, key=lambda k: cv_rmse_by_k[k])
q8_test_rmse = knn_rmse(q8_best_k)
