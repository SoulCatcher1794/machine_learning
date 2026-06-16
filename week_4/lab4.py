# Grade Cell: Import Libraries
#
# This cell imports all necessary libraries for the assignment.
from typing import Tuple, List

import numpy as np
import pandas as pd

# Plotting is optional; imported lazily if needed
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Set a random state for reproducibility
RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)



# Grade Cell: Question 1
#
# Task: Load the dataset and display its first 5 rows.
#
# Instructions:
# 1. Load the 'diabetes.csv' file into a pandas DataFrame called `df`.
# 2. Use the `.head()` method to display the first 5 rows.

df = pd.read_csv('diabetes.csv')
df.head()



# Grade Cell: Question 2
#
# Task: Prepare the data for modeling.
#
# Instructions:
# 1. Create the feature matrix `X` with all columns except `target`.
# 2. Create the target vector `y` from the `target` column.

X = df.drop(columns='target')
y = df['target']




# Grade Cell: Question 3
#
# Task: Split and scale the data.
#
# Instructions:
# 1. Split `X` and `y` into `X_train`, `X_test`, `y_train`, and `y_test` with a `test_size` of 0.2 and `random_state=RANDOM_STATE`.
# 2. Initialize a `StandardScaler` and fit it on `X_train`.
# 3. Transform both `X_train` and `X_test` using the fitted scaler, naming them `X_train_scaled` and `X_test_scaled`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Grade Cell: Question 4
#
# Task: Train a baseline OLS model and compute metrics.
#
# Instructions:
# 1. Fit `LinearRegression` using `X_train_scaled`, `y_train`.
# 2. Compute test RMSE (`rmse_ols_test`) and R^2 (`r2_ols_test`).

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
rmse_ols_test = round(float(np.sqrt(mean_squared_error(y_test,y_pred))), 3)
r2_ols_test = round(float(r2_score(y_test,y_pred)), 3)



# Grade Cell: Question 5
#
# Task: Compute CV RMSE mean and std for OLS on the training split.
#
# Instructions:
# - Use `cross_val_score` with `scoring='neg_mean_squared_error'` and `KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`.
# - Store `rmse_ols_cv_mean` and `rmse_ols_cv_std` (floats).

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_neg_mse = cross_val_score(lr, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_neg_mse)

rmse_ols_cv_mean = float(np.mean(cv_rmse))
rmse_ols_cv_std = float(np.std(cv_rmse))

# Grade Cell: Question 6
#
# Task: Train RidgeCV and compute test metrics.
#
# Instructions:
# - Use `ridge_alphas = np.logspace(-3, 3, 50)`
# - Fit on `X_train_scaled`, `y_train`
# - Store `ridge_best_alpha`, `rmse_ridge_test`, `r2_ridge_test`

ridge_alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=ridge_alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)
ridge_best_alpha = ridge_cv.alpha_
y_pred_ridge = ridge_cv.predict(X_test_scaled)
rmse_ridge_test = round(float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))), 3)
r2_ridge_test = round(float(r2_score(y_test, y_pred_ridge)), 3)



# Grade Cell: Question 7
#
# Task: Train LassoCV and compute metrics.
#
# Instructions:
# - Use `lasso_alphas = np.logspace(-3, 1, 50)`, `cv=5`, `random_state=RANDOM_STATE`, `max_iter=10000`
# - Store `lasso_best_alpha`, `rmse_lasso_test`, `r2_lasso_test`, `n_nonzero_lasso`

lasso_alphas = np.logspace(-3, 1, 50)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=5, random_state=RANDOM_STATE, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
lasso_best_alpha = lasso_cv.alpha_
y_pred_lasso = lasso_cv.predict(X_test_scaled)
rmse_lasso_test = round(float(np.sqrt(mean_squared_error(y_test, y_pred_lasso))), 3)
r2_lasso_test = round(float(r2_score(y_test, y_pred_lasso)), 3)
n_nonzero_lasso = int(np.sum(lasso_cv.coef_ != 0))



# Grade Cell: Question 8
#
# Task: Expose arrays for CV curves.
#
# Instructions:
# - For Ridge: use `ridge_alphas` and compute mean MSE over folds from `ridge_cv.cv_values_` (if available).
# - For Lasso: use `lasso_alphas` and `lasso_cv.mse_path_` (mean across folds).


ridge_cv_mse_mean = np.mean(ridge_cv.cv_values_, axis=0)
lasso_cv_mse_mean = np.mean(lasso_cv.mse_path_, axis=1)


# Grade Cell: Question 9
#
# Task: Implement a bootstrap procedure for OLS coefficients.
#
# Instructions:
# - Implement a function `bootstrap_ols_coefficients` that:
#   * draws B bootstrap samples of the training set (with replacement)
#   * fits OLS on each sample using scaled features
#   * stores coefficient vectors
#   * returns (coef_bootstrap_df, coef_ci_95)
# - Use B=200, RANDOM_STATE

def bootstrap_ols_coefficients(X_train_scaled, y_train, B, random_state):
    rng = np.random.default_rng(random_state)

    if isinstance(X_train_scaled, pd.DataFrame):
        X_values = X_train_scaled.to_numpy()
        feature_names = list(X_train_scaled.columns)
    else:
        X_values = np.asarray(X_train_scaled)
        if "X" in globals() and hasattr(X, "columns") and len(X.columns) == X_values.shape[1]:
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

    y_values = np.asarray(y_train)

    n_samples, n_features = X_values.shape
    coef_bootstrap = np.zeros((B, n_features))

    for iter in range(B):
        idx = rng.choice(n_samples, size=n_samples, replace=True) # Get bootstrap sample indices
        X_bootstrap = X_values[idx] # Get bootstrap sample features
        y_bootstrap = y_values[idx] # Get bootstrap sample targets

        lr = LinearRegression()
        lr.fit(X_bootstrap, y_bootstrap) # Fit OLS on bootstrap sample
        coef_bootstrap[iter] = lr.coef_ # Store coefficients for each bootstrap iteration

    coef_bootstrap_df = pd.DataFrame(coef_bootstrap, columns=feature_names) # Convert to DataFrame for quantile calculation
    coef_ci_95 = pd.DataFrame({
        "lower": coef_bootstrap_df.quantile(0.025),
        "upper": coef_bootstrap_df.quantile(0.975),
    }, index=feature_names) # Compute 95% confidence intervals for each coefficient
    return coef_bootstrap_df, coef_ci_95

coef_bootstrap_df, coef_ci_95 = bootstrap_ols_coefficients(
    X_train_scaled,
    y_train,
    B=200,
    random_state=RANDOM_STATE,
)



# Grade Cell: Question 10
#
# Task: Implement Bootstrap OOB RMSE for OLS.
#
# Instructions:
# - Implement `bootstrap_oob_rmse_ols` that returns (rmse_oob_mean, rmse_oob_ci95)
# - Use B=200, RANDOM_STATE

def bootstrap_oob_rmse_ols(X_train_scaled, y_train, B, random_state):
    rng = np.random.default_rng(random_state)

    n_samples = X_train_scaled.shape[0]
    rmse_oob = np.zeros(B)

    for iter in range(B):
        idx = rng.choice(n_samples, size=n_samples, replace=True) # Get bootstrap sample indices
        oob_idx = np.setdiff1d(np.arange(n_samples), idx) # Get out-of-bag indices

        if len(oob_idx) == 0: # If no OOB samples, skip this iteration
            continue

        X_bootstrap = X_train_scaled[idx] # Get bootstrap sample features
        y_bootstrap = y_train.iloc[idx] # Get bootstrap sample targets

        lr = LinearRegression()
        lr.fit(X_bootstrap, y_bootstrap) # Fit OLS on bootstrap sample

        X_oob = X_train_scaled[oob_idx] # Get OOB features
        y_oob = y_train.iloc[oob_idx] # Get OOB targets
        y_oob_pred = lr.predict(X_oob) # Predict on OOB samples
        rmse_oob[iter] = np.sqrt(mean_squared_error(y_oob, y_oob_pred)) # Compute OOB RMSE for this iteration

    rmse_oob_mean = round(float(np.mean(rmse_oob)), 3)
    rmse_oob_ci95 = (round(float(np.percentile(rmse_oob, 2.5)), 3), round(float(np.percentile(rmse_oob, 97.5)), 3))
    return rmse_oob_mean, rmse_oob_ci95

rmse_oob_mean, rmse_oob_ci95 = bootstrap_oob_rmse_ols(
    X_train_scaled,
    y_train,
    B=200,
    random_state=RANDOM_STATE,
)
