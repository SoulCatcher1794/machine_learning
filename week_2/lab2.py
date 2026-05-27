import pathlib
from typing import Tuple, Dict, List

import numpy as npm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE: int = 42  # global seed for full reproducibility
np.random.seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("auto.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "auto.csv is missing from the lab directory. Please download it or ask the TA "
        "for assistance."
    )

# Part 1: Data loading
def load_auto_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    return df

# Compute the answer required by the autograder
q1_shape: Tuple[int, int] = load_auto_data().shape

#Part 2: Prepare data
df = load_auto_data()
X = df[["displacement","horsepower","weight","acceleration"]]
y = df["mpg"]

q2_num_features = X.shape[1]

# Part 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE)
n_train = len(X_train)
n_test = len(X_test)
q3_split_counts = (n_train, n_test)

# Part 4: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Part 5: Calculate Test MSE
q5_mse = round(mean_squared_error(y_test, y_pred), 3)

# Part 6: Calculate Test R²
q6_r2 = round(r2_score(y_test, y_pred), 3)

# Part 7: Reusable Evaluation Function

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = float(round(mean_squared_error(y_test, y_pred), 3))
    r2 = float(round(r2_score(y_test, y_pred), 3))
    eval = (mse, r2)
    return eval

# Part 8: Analyze Coefficients
coefficients = pd.Series(model.coef_, index=X_train.columns)
q8_strongest_negative_feature = coefficients.idxmin()