import pathlib
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Suppress TensorFlow warnings for cleaner output
tf.get_logger().setLevel("ERROR")

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("breast_cancer_nn.csv")
if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "breast_cancer_nn.csv is missing from the lab directory. Please download it or ask the TA "
        "for assistance."
    )

# Question 1: Load the dataset and check its shape

def load_data() -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset from CSV.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the 30 features plus the target column.
    """
    return pd.read_csv(_DATA_PATH)

# Compute the answer required by the autograder
q1_shape: Tuple[int, int] = load_data().shape

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
print(f"Dataset shape: {q1_shape}")



# Question 2: Separate features and target

df = load_data()

X = df.drop(columns="target")
y = df["target"]
q2_n_features = X.shape[1]

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(q2_n_features, (int, np.integer)), (
    "q2_n_features must be an integer. Use X.shape[1] to get the number of columns."
)
assert q2_n_features > 0, (
    "Number of features must be positive. Did you create X correctly?"
)
assert "target" not in X.columns, (
    "The target column 'target' should not be in your feature matrix X. "
    "Use df.drop(columns='target') to remove it."
)
print(f"Number of features: {q2_n_features}")


# Question 3: Split the dataset into training, validation, and test sets

# First split into train+val (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Then split train+val into train (75%) and val (25%).
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val, 
    train_size=0.75,
    random_state=RANDOM_STATE
)

# Create standard scaler instance
sc = StandardScaler()
# Fit only on training data, transform training, validation, and test data
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)
X_test_scaled = sc.transform(X_test)

# Store row counts for each data split
n_train = X_train_scaled.shape[0]
n_val = X_val_scaled.shape[0]
n_test = X_test_scaled.shape[0]
q3_split_counts = (n_train, n_val, n_test)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert isinstance(q3_split_counts, tuple), (
    "q3_split_counts must be a tuple like (n_train, n_val, n_test)."
)
assert len(q3_split_counts) == 3, "Tuple should have exactly 3 elements."
n_train, n_val, n_test = q3_split_counts
assert all(n > 0 for n in q3_split_counts), (
    "All splits must have positive row counts."
)
assert n_train > n_val and n_train > n_test, (
    "Training set should be the largest. Check your split ratios."
)
print(f"Train/Val/Test split: {n_train} / {n_val} / {n_test}")



# Question 4: Build a Neural network model using Tensorflow Keras

model = tf.keras.Sequential([
    tf.keras.Input(shape=(q2_n_features,)), # Define input shape
    tf.keras.layers.Dense(16, activation='relu'), # First layer, 16 neurons, ReLU activation
    tf.keras.layers.Dense(8, activation='relu'), # Second layer, 8 neurons, ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer, 1 neuron, sigmoid activation
])

# Count the number of trainable parameters in the model    
q4_n_params = model.count_params()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert isinstance(model, Sequential), (
    "model should be a Keras Sequential model. "
    "Did you use Sequential([...])?"
)
assert isinstance(q4_n_params, (int, np.integer)), (
    "q4_n_params must be an integer. Use model.count_params()."
)
assert q4_n_params > 0, (
    "Model should have trainable parameters. Did you add layers?"
)
assert len(model.layers) == 3, (
    "Model should have exactly 3 layers (2 hidden + 1 output). "
    "Check your architecture."
)
print(f"Total trainable parameters: {q4_n_params}")



# Question 5: Compile and train the model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert hasattr(history, 'history'), (
    "history should be a Keras History object. Did you call model.fit()?"
)
assert 'accuracy' in history.history, (
    "History should contain 'accuracy'. Did you include metrics=['accuracy'] in compile?"
)
assert 'val_accuracy' in history.history, (
    "History should contain 'val_accuracy'. Did you pass validation_data to fit()?"
)
assert isinstance(q5_final_train_acc, float), (
    "q5_final_train_acc must be a float."
)
assert 0 <= q5_final_train_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
print(f"Final training accuracy: {q5_final_train_acc:.3f}")