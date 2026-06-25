# Grade Cell: Import Libraries
#
# This cell imports all necessary libraries for the assignment.
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# Set a random state for reproducibility
RANDOM_STATE: int = 42

# Grade Cell: Question 1
#
# Task: Load the datasets and show a preview.
#
# Instructions:
# 1. Load the 'diabetes_regression.csv' into DataFrame `df_reg`.
# 2. Load the 'breast_cancer_classification.csv' into DataFrame `df_cls`.
# 3. Display the first 5 rows of each.

# Load datasets
df_reg = pd.read_csv("diabetes_regression.csv")
df_cls = pd.read_csv("breast_cancer_classification.csv")

# Display the first 5 rows of each DataFrame
print(df_reg.head())
print(df_cls.head())

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert "df_reg" in locals(), "DataFrame 'df_reg' not found."
assert "df_cls" in locals(), "DataFrame 'df_cls' not found."
assert df_reg.shape[1] >= 2, "df_reg should include features and target."
assert df_cls.shape[1] >= 2, "df_cls should include features and target."
assert "target" in df_reg.columns, "df_reg must contain 'target' column."
assert "target" in df_cls.columns, "df_cls must contain 'target' column."
assert df_reg.isnull().sum().sum() == 0, "df_reg should have no missing values."
assert df_cls.isnull().sum().sum() == 0, "df_cls should have no missing values."
print("Datasets loaded successfully.")
df_reg.head()
df_cls.head()



# Grade Cell: Question 2
#
# Task: Train a baseline regression tree and compute metrics.
#
# Instructions:
# 1. Split df_reg into features X_reg and target y_reg; then into train/test with test_size=0.25 and random_state=RANDOM_STATE.
# 2. Fit DecisionTreeRegressor(random_state=RANDOM_STATE) as tree_reg.
# 3. Compute train and test metrics: MSE, MAE, and R^2.

# Create features and target for regression
X_reg = df_reg.drop(columns=["target"])
y_reg = df_reg["target"]

# Train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.25, random_state=RANDOM_STATE)

# Train regression tree and calculate predictions
tree_reg = DecisionTreeRegressor(random_state=RANDOM_STATE)
tree_reg.fit(X_train_reg, y_train_reg)
y_train_pred = tree_reg.predict(X_train_reg)
y_test_pred = tree_reg.predict(X_test_reg)

# Compute metrics for training set
mse_tree_train = mean_squared_error(y_train_reg, y_train_pred)
mae_tree_train = mean_absolute_error(y_train_reg, y_train_pred)
r2_tree_train = r2_score(y_train_reg, y_train_pred)

# Compute metrics for test set
mse_tree_test = mean_squared_error(y_test_reg, y_test_pred)
mae_tree_test = mean_absolute_error(y_test_reg, y_test_pred)
r2_tree_test = r2_score(y_test_reg, y_test_pred)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert "tree_reg" in locals(), "Baseline regression tree not found."
assert hasattr(tree_reg, "tree_"), "Model does not appear to be trained."
assert mse_tree_train >= 0.0 and mse_tree_test >= 0.0, "MSE should be non-negative."
assert -1.0 <= r2_tree_test <= 1.0, "R^2 should be between -1 and 1."
print(
    f"Train MSE: {mse_tree_train:.2f}, Test MSE: {mse_tree_test:.2f}, Test MAE: {mae_tree_test:.2f}, Test R2: {r2_tree_test:.3f}"
)

# Grade Cell: Question 3
#
# Task: Depth sweep on regression tree.
#
# Instructions:
# 1. For `depths = [1, 2, 3, 4, 5, 6, 8, 10, None]`, fit a tree for each depth.
# 2. Record train/test MSE in lists `train_mse_per_depth` and `test_mse_per_depth` aligned with `depths`.
# 3. Compute `best_depth` as the depth (value from the list) with minimal test MSE.

# Define depths to evaluate
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]

# Initialize lists to store MSE values
train_mse_per_depth = []
test_mse_per_depth = []

# Fit trees for each depth and compute MSE
for depth in depths:
    # Fit a DecisionTreeRegressor with the specified depth
    tree = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(X_train_reg, y_train_reg)

    # Compute predictions for training and test sets
    y_train_pred = tree.predict(X_train_reg)
    y_test_pred = tree.predict(X_test_reg)

    # Calculate and store MSE values
    train_mse_per_depth.append(mean_squared_error(y_train_reg, y_train_pred))
    test_mse_per_depth.append(mean_squared_error(y_test_reg, y_test_pred))

# Determine the best depth based on minimal test MSE
best_depth_value = min(test_mse_per_depth)
# Find the corresponding depth from the depths list using the index of the best test MSE
best_depth = depths[test_mse_per_depth.index(best_depth_value)]

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert len(train_mse_per_depth) == len(depths), "train_mse_per_depth length mismatch."
assert len(test_mse_per_depth) == len(depths), "test_mse_per_depth length mismatch."
assert isinstance(best_depth, int), "best_depth must be an int."
assert best_depth > 0, "best_depth must be positive."
print("Depths:", depths)
print("Test MSE per depth:", [round(v, 2) for v in test_mse_per_depth])
print("Best depth:", best_depth)

# Grade Cell: Question 4
#
# Task: Prune via cost-complexity path and CV.
#
# Instructions:
# 1. Fit an unpruned tree on training data and obtain `ccp_alphas` via `cost_complexity_pruning_path` on the TRAIN split only.
# 2. Subsample up to 20 unique alphas, evenly spaced.
# 3. For each alpha, perform 5-fold CV on the TRAIN split to estimate MSE.
# 4. Choose `alpha_hat` with minimal CV MSE; fit `tree_pruned` with that alpha and evaluate `mse_pruned` and `r2_pruned` on TEST.

# Fit an unpruned regression tree on the training data
unpruned_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
unpruned_tree.fit(X_train_reg, y_train_reg)

# Obtain the cost-complexity pruning path and extract ccp_alphas
ccp_alphas = unpruned_tree.cost_complexity_pruning_path(X_train_reg, y_train_reg).ccp_alphas

# Create a subsample of up to 20 unique alphas, evenly spaced
# Use the minimum of 20 and the length of ccp_alphas to ensure we don't exceed available alphas
alphas_sorted = np.linspace(ccp_alphas.min(), ccp_alphas.max(), num=min(20, len(ccp_alphas))) 

# Define a list to store the mean squared error for each alpha
mse_per_alpha = []
# Define a KFold cross-validator with 5 splits, shuffling, and a fixed random state for reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Perform cross-validation for each alpha to estimate MSE
for alpha in alphas_sorted:
    # Fit a DecisionTreeRegressor with the current alpha
    tree = DecisionTreeRegressor(random_state=RANDOM_STATE, ccp_alpha=alpha)
    # Initialize a list to store MSE scores for the current alpha across folds
    mse_scores = []

    # Get the train and validation indices for each fold
    for train_index, test_index in kf.split(X_train_reg):
        # Split the training data into the current fold's training and validation sets
        X_train_fold = X_train_reg.iloc[train_index]
        X_test_fold = X_train_reg.iloc[test_index] 
        y_train_fold = y_train_reg.iloc[train_index]
        y_test_fold = y_train_reg.iloc[test_index]

        # Fit the tree on the current fold's training data
        tree.fit(X_train_fold, y_train_fold)
        y_test_pred = tree.predict(X_test_fold)

        # Compute the MSE for the current fold and append it to the list of scores
        mse_scores.append(mean_squared_error(y_test_fold, y_test_pred))

    # Compute the mean MSE across all folds for the current alpha and append it to the list of MSEs
    mse_per_alpha.append(np.mean(mse_scores))

# Determine the best alpha that minimizes the mean MSE across folds
best_mse = min(mse_per_alpha)
best_alpha_index = mse_per_alpha.index(best_mse)
alpha_hat = alphas_sorted[best_alpha_index]

# Fit the pruned tree with the best alpha and evaluate on the test set
tree_pruned = DecisionTreeRegressor(random_state=RANDOM_STATE, ccp_alpha=alpha_hat)
tree_pruned.fit(X_train_reg, y_train_reg)
y_test_pred = tree_pruned.predict(X_test_reg)

# Calculate metrics for the pruned tree
mse_pruned = mean_squared_error(y_test_reg, y_test_pred)
r2_pruned = r2_score(y_test_reg, y_test_pred)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4

min_alpha = np.float64(0.0)
max_alpha = np.float64(1755.6938868563993)

assert "alpha_hat" in locals(), "alpha_hat not found."
assert isinstance(alpha_hat, float), "alpha_hat should be a float."
assert alpha_hat >= 0, "alpha_hat must be non-negative."
assert mse_pruned >= 0.0, "mse_pruned should be non-negative."
assert -1.0 <= r2_pruned <= 1.0, "r2_pruned should be between -1 and 1."
print(
    f"alpha_hat={alpha_hat:.2e}, Pruned Test MSE={mse_pruned:.2f}, R2={r2_pruned:.3f}"
)

print(alphas_sorted.max())



# Grade Cell: Question 5
#
# Task: Train classification tree and evaluate.
#
# Instructions:
# 1. Split df_cls into X_cls, y_cls; then stratified train/test with test_size=0.25, random_state=RANDOM_STATE.
# 2. Fit DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE) as tree_cls.
# 3. Compute `acc_tree_cls` and `conf_mat_tree_cls` on TEST.

# Create features and target for classification
X_cls = df_cls.drop(columns=["target"])
y_cls = df_cls["target"]

# Perform stratified train/test split for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, 
    y_cls, 
    test_size=0.25, 
    random_state=RANDOM_STATE, 
    stratify=y_cls # Ensure class distribution is preserved in train/test split
)

# Fit the classification tree and make predictions on the test set
tree_cls = DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE)
tree_cls.fit(X_train_cls, y_train_cls)
y_cls_pred = tree_cls.predict(X_test_cls)

# Compute metrics for the classification tree
acc_tree_cls = accuracy_score(y_test_cls, y_cls_pred)
conf_mat_tree_cls = confusion_matrix(y_test_cls, y_cls_pred)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert "tree_cls" in locals(), "Classification tree not found."
assert conf_mat_tree_cls.shape == (2, 2), "Confusion matrix must be 2x2."
assert 0.0 <= acc_tree_cls <= 1.0, "Accuracy must be between 0 and 1."
print(f"Classification Accuracy: {acc_tree_cls:.3f}")
print("Confusion Matrix:\n", conf_mat_tree_cls)



# Grade Cell: Question 6
#
# Task: Compute Gini and information gain for candidate splits.
#
# Instructions:
# 1. Generate a small 2D dataset with two classes (balanced-ish).
# 2. For a few split thresholds on x1 (e.g., -0.5, 0.0, 0.5), compute Gini(parent), Gini(left), Gini(right), and information gain.
# 3. Store gains in list `info_gain_values` aligned with `splits_x1` and show that the "middle" split has higher gain than edge splits.

# Create a 2D (2 features) dataset with two classes for testing
test_dataset = pd.DataFrame({
    "x1": [-1, -0.5, 0, 0.5, 1, 1.5, 2], # Feature values
    "x2": [0, 0, 0, 0, 0, 0, 0], # Second feature (not used in splits)
    "target": [1, 0, 1, 0, 1, 0, 0] # Class labels (binary target)
})

# Define candidate split thresholds for x1 according to suggested values
splits_x1 = [-0.5, 0.0, 0.5]

# Initialize a list to store information gain values for each split
info_gain_values = []

# Define a function to compute Gini impurity for a given class label series
def gini_impurity(y):
    # If the series is empty, return 0.0 as the impurity
    if len(y) == 0:
        return 0.0
    # Calculate the proportion of each class in the series
    p = y.value_counts(normalize=True)
    # Compute Gini impurity using the formula: Gini = 1 - sigma(p_i^2) where p_i is the proportion of class i
    gini = 1.0 - sum(p**2) # This contains the sum of squared proportions for each class
    return gini

for split in splits_x1:
    # Define left and right splits based on the current threshold
    left_split = test_dataset[test_dataset["x1"] < split]
    right_split = test_dataset[test_dataset["x1"] >= split]

    # Compute Gini impurity for the parent, left, and right splits
    gini_parent = gini_impurity(test_dataset["target"])
    gini_left = gini_impurity(left_split["target"])
    gini_right = gini_impurity(right_split["target"])

    # Define lengths of left and right splits for weighted Gini calculation
    len_left = len(left_split)
    len_right = len(right_split)
    len_total = len(test_dataset)

    # Compute information gain for the current split
    info_gain = gini_parent - ((len_left/len_total)*gini_left + (len_right/len_total)*gini_right)
    info_gain_values.append(info_gain)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 6
assert len(info_gain_values) == 3, "Expect three info gain values."
assert all(
    isinstance(v, float) for v in info_gain_values
), "All gains should be floats."
assert all(
    v >= 0 for v in info_gain_values
), "Information gains should be non-negative."
print(
    "Information gains (x1 splits at -0.5, 0.0, 0.5):",
    [round(v, 3) for v in info_gain_values],
)



# Grade Cell: Question 7
#
# Task: Bagging for regression.
#
# Instructions:
# 1. Fit `BaggingRegressor(DecisionTreeRegressor(random_state=RANDOM_STATE), n_estimators=200, bootstrap=True, oob_score=True, random_state=RANDOM_STATE)`.
# 2. Compute `mse_bag` on TEST and record `oob_r2_bag`.
#
# Use the output splits generated by split_regression_data() in Question 2.

# Fit BaggingRegressor with DecisionTreeRegressor as base estimator
bag_reg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
    n_estimators=200,
    bootstrap=True, # Sampling with replacement
    oob_score=True, # Calculate out-of-bag score for R^2
    random_state=RANDOM_STATE
)

# Fit the bagging regressor on the training data
bag_reg.fit(X_train_reg, y_train_reg)

# Calculate metrics on the test set
mse_bag = mean_squared_error(y_test_reg, bag_reg.predict(X_test_reg))
# This is enabled by setting oob_score=True during initialization
oob_r2_bag = bag_reg.oob_score_ 

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert "bag_reg" in locals(), "BaggingRegressor not found."
assert 0.0 <= bag_reg.n_estimators <= 500, "n_estimators must be <= 500."
assert mse_bag >= 0.0, "mse_bag should be non-negative."
assert -1.0 <= oob_r2_bag <= 1.0, "OOB R^2 must be between -1 and 1."
print(f"Bagging: Test MSE={mse_bag:.2f}, OOB R2={oob_r2_bag:.3f}")



# Grade Cell: Question 8
#
# Task: Random Forest sweep.
#
# Instructions:
# 1. For `max_features` in `[1.0, 0.5, 'sqrt']`, fit a `RandomForestRegressor(n_estimators=300, bootstrap=True, oob_score=True, random_state=RANDOM_STATE)`.
# 2. Store for each setting: test MSE (`mse`), OOB R^2 (`oob_r2`), and feature importances (`importances`).
# 3. Save results in dict `rf_results` keyed by the `max_features` value (as str), each value a dict with keys `'mse'`, `'oob_r2'`, `'importances'`.
# 4. Extract top-10 feature indices for `'sqrt'` setting into list `top10_features_rf` (descending importance).
#
# Use the output splits generated by split_regression_data() in Question 2.

# Define a dictionary to store results for each max_features setting
rf_results = {}


# Define the list of max_features settings to evaluate
# This indicates the number of features to consider when looking for the best split
max_features = [1.0, 0.5, 'sqrt']

# Define a function to perform Random Forest regression and return metrics
def rf_regression(X_train, y_train, X_test, y_test, max_feature):
    # Initialize the RandomForestRegressor with specified parameters
    rf = RandomForestRegressor(
        n_estimators=300,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        max_features=max_feature
    )
    
    # Train the Random Forest model on the training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Compute metrics on the test set
    mse = mean_squared_error(y_test, y_pred)
    oob_r2 = rf.oob_score_
    # These are the feature importances computed by the Random Forest model
    importances = rf.feature_importances_

    # Store results in the rf_results dictionary with max_feature as the key
    rf_results[str(max_feature)] = {
        "mse": mse,
        "oob_r2": oob_r2,
        "importances": importances
    }

# Perform Random Forest regression for each max_features setting
for mf in max_features:
    rf_regression(X_train_reg, y_train_reg, X_test_reg, y_test_reg, mf)
    
# Extract the importances for the 'sqrt' setting
importances_sqrt = rf_results["sqrt"]["importances"]
# Sort the indices of the importances in ascending order [::-1] and extract the top 10 most important features [:10]
top10_features_rf = np.argsort(importances_sqrt)[::-1][:10]

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
for key in ["1.0", "0.5", "sqrt"]:
    assert key in rf_results, f"Missing RF result for {key}."
    assert "mse" in rf_results[key] and "oob_r2" in rf_results[key], "Missing metrics."
    assert -1.0 <= rf_results[key]["oob_r2"] <= 1.0, "OOB R^2 out of bounds."
    assert rf_results[key]["mse"] >= 0, "MSE must be non-negative."
assert len(top10_features_rf) == 10, "Expect top10 feature indices."
print("RF (sqrt) top-10 features:", top10_features_rf)


# Grade Cell: Question 9
#
# Task: Boosting with validation selection.
#
# Instructions:
# 1. Create a train/validation split from the original training set (e.g., 30% as validation; random_state=RANDOM_STATE).
# 2. For learning rates in `[1.0, 0.1, 0.03]`, fit `GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=RANDOM_STATE)` and record train and validation MSE over staged predictions.
# 3. Determine `best_params_gbr` as a dict with keys `learning_rate` and `best_iter` (iteration minimizing validation MSE).
# 4. Fit a final model with that learning rate and `n_estimators=best_iter` on the full training set and compute `mse_gbr_test` on TEST.
#
# Use the output splits generated by split_regression_data() in Question 2.

X_train_boost, X_val_boost, y_train_boost, y_val_boost = train_test_split(X_train_reg, y_train_reg, test_size=0.3, random_state=RANDOM_STATE)
learning_rates = [1.0, 0.1, 0.03]
gbr_results = {}
best_params_gbr = {}
best_val_mse_overall = float('inf')

for lr in learning_rates:
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=1, learning_rate=lr, random_state=RANDOM_STATE)
    gbr.fit(X_train_boost, y_train_boost)

    train_mse = []
    val_mse = []

    for y_train_pred in gbr.staged_predict(X_train_boost):
        train_mse.append(mean_squared_error(y_train_boost, y_train_pred))

    for y_val_pred in gbr.staged_predict(X_val_boost):
        val_mse.append(mean_squared_error(y_val_boost, y_val_pred))

    gbr_results[lr] = {
        "train_mse": train_mse,
        "val_mse": val_mse
    }



for lr, mse_dict in gbr_results.items():
    best_iter_for_lr = int(np.argmin(mse_dict["val_mse"]) + 1)
    best_val_mse_for_lr = mse_dict["val_mse"][best_iter_for_lr - 1]

    if best_val_mse_for_lr < best_val_mse_overall:
        best_val_mse_overall = best_val_mse_for_lr
        best_params_gbr = {
            "learning_rate": lr,
            "best_iter": best_iter_for_lr
        }

final_gbr = GradientBoostingRegressor(
    n_estimators=best_params_gbr["best_iter"], 
    max_depth=1, 
    learning_rate=best_params_gbr["learning_rate"], 
    random_state=RANDOM_STATE
)

final_gbr.fit(X_train_reg, y_train_reg)
mse_gbr_test = mean_squared_error(y_test_reg, final_gbr.predict(X_test_reg))

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert "gbr_results" in locals(), "gbr_results missing."
assert isinstance(best_params_gbr, dict), "best_params_gbr must be a dict."
assert (
    "learning_rate" in best_params_gbr and "best_iter" in best_params_gbr
), "best_params_gbr missing keys."
assert isinstance(best_params_gbr["best_iter"], int), "best_iter must be an integer."
assert best_params_gbr["best_iter"] > 0, "best_iter must be positive."
assert mse_gbr_test >= 0.0, "mse_gbr_test should be non-negative."
print("Best GBR params:", best_params_gbr)
print(f"GBR Test MSE: {mse_gbr_test:.2f}")



# Grade Cell: Question 10
#
# Task: Global and local interpretation on classification.
#
# Instructions:
# 1. Fit `RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=RANDOM_STATE)` and `GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, max_depth=1, random_state=RANDOM_STATE)`.
# 2. Compute permutation importance on the TEST set for both; store arrays in `perm_importance_rf` and `perm_importance_gb`.
# 3. Select the top-1 feature index from RF, store as `pdp_feature_index` and generate PDP+ICE plots for that feature for RF and GBM (plots not graded).
#
# Use the output splits generated by split_classification_data() in Question 5."

rf_cls = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=RANDOM_STATE)
rf_cls.fit(X_train_cls, y_train_cls)

gb_cls = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, max_depth=1, random_state=RANDOM_STATE)
gb_cls.fit(X_train_cls, y_train_cls)

perm_importance_rf = permutation_importance(rf_cls, X_test_cls, y_test_cls, n_repeats=10, random_state=RANDOM_STATE).importances_mean
perm_importance_gb = permutation_importance(gb_cls, X_test_cls, y_test_cls, n_repeats=10, random_state=RANDOM_STATE).importances_mean

pdp_feature_index = np.argmax(perm_importance_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

PartialDependenceDisplay.from_estimator(
    rf_cls, X_test_cls, features=[pdp_feature_index], 
    kind='both', ax=axes[0], grid_resolution=50
)
axes[0].set_title(f'RF: PDP+ICE for Feature {pdp_feature_index}')

PartialDependenceDisplay.from_estimator(
    gb_cls, X_test_cls, features=[pdp_feature_index], 
    kind='both', ax=axes[1], grid_resolution=50
)

axes[1].set_title(f'GB: PDP+ICE for Feature {pdp_feature_index}')

plt.tight_layout()
plt.show()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 10
assert (
    "perm_importance_rf" in locals() and "perm_importance_gb" in locals()
), "Permutation importances not found."
assert (
    perm_importance_rf.ndim == 1 and perm_importance_gb.ndim == 1
), "Importances should be 1D arrays."
assert 0 <= pdp_feature_index < X_train_cls.shape[1], "pdp_feature_index out of bounds."
assert isinstance(
    pdp_feature_index, (int, np.integer)
), "pdp_feature_index must be an integer."
print("Top feature index (RF, perm importance):", pdp_feature_index)