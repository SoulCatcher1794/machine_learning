import pathlib
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TensorFlow/Keras imports
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress TensorFlow warnings for cleaner output
tf.get_logger().setLevel("ERROR")

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Fashion-MNIST class names for reference
CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

_TRAIN_PATH = pathlib.Path("fashion_mnist_train.csv")
_TEST_PATH = pathlib.Path("fashion_mnist_test.csv")

if not _TRAIN_PATH.exists() or not _TEST_PATH.exists():
    raise FileNotFoundError(
        "Fashion-MNIST CSV files are missing from the lab directory. "
        "Please run the download script or ask the TA for assistance."
    )

# Question 1: Load and Explore the Dataset

def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load a Fashion-MNIST CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing pixel values and labels.
    """
    return pd.read_csv(path)


# Load training data and compute answer
train_df = load_data(_TRAIN_PATH)
q1_shape: Tuple[int, int] = train_df.shape

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 1
assert isinstance(q1_shape, tuple), (
    "q1_shape must be a tuple. Use df.shape which returns (rows, cols)."
)
assert len(q1_shape) == 2, (
    "q1_shape should have 2 elements (rows, cols)."
)
assert q1_shape[0] > 0 and q1_shape[1] > 0, (
    "Shape values must be positive. Is your CSV loading correctly?"
)
print(f"Training dataset shape: {q1_shape}")



# Question 2: Preprocess the Data

# Load test data as well
test_df = load_data(_TEST_PATH)

# Extract features and labels from the training dataset
# Normalize pixel info to be in the range [0, 1] by dividing by 255
# Transform to Numpy arrays so min() and max() operations return a single scalar value
X = train_df.drop(columns="label").to_numpy() / 255.0
y = train_df["label"].to_numpy()

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

# Calculate size of each dataset split
n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

q2_split_counts = (n_train, n_val, n_test)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 2
assert isinstance(q2_split_counts, tuple), (
    "q2_split_counts must be a tuple (n_train, n_val, n_test)."
)
assert len(q2_split_counts) == 3, "Tuple should have exactly 3 elements."
n_train, n_val, n_test = q2_split_counts
assert all(n > 0 for n in q2_split_counts), (
    "All splits must have positive sample counts."
)
assert n_train > n_val and n_train > n_test, (
    "Training set should be the largest. Check your split ratios."
)
assert X_train.max() <= 1.0 and X_train.min() >= 0.0, (
    "Pixel values should be normalized to [0, 1]. Did you divide by 255?"
)
print(f"Train/Val/Test split: {n_train} / {n_val} / {n_test}")



# Question 3: Build a Baseline Model (No Regularization)

# Create the baseline model
model_baseline = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)), # Input layer = Number of features
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the baseline model
model_baseline.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Obtain the number of parameters
q3_n_params = model_baseline.count_params()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 3
assert isinstance(model_baseline, Sequential), (
    "model_baseline should be a Keras Sequential model."
)
assert isinstance(q3_n_params, (int, np.integer)), (
    "q3_n_params must be an integer. Use model.count_params()."
)
assert len(model_baseline.layers) == 4, (
    "Model should have 4 layers (3 hidden + 1 output)."
)
assert q3_n_params > 200000, (
    "This large model should have many parameters. Check your layer sizes."
)
print(f"Baseline model parameters: {q3_n_params:,}")



# Question 4: Train the Baseline Model

# Train the baseline model
history_baseline = model_baseline.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    verbose=0,
    validation_data=(X_val, y_val)
)

# Obtain the final validation accuracy
q4_final_val_acc = round(float(history_baseline.history["val_accuracy"][-1]), 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 4
assert hasattr(history_baseline, 'history'), (
    "history_baseline should be a Keras History object. Did you call model.fit()?"
)
assert 'accuracy' in history_baseline.history, (
    "History should contain 'accuracy'. Did you include metrics=['accuracy']?"
)
assert 'val_accuracy' in history_baseline.history, (
    "History should contain 'val_accuracy'. Did you pass validation_data?"
)
assert isinstance(q4_final_val_acc, float), (
    "q4_final_val_acc must be a float."
)
assert 0 <= q4_final_val_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
print(f"Final validation accuracy (baseline): {q4_final_val_acc:.3f}")



# Question 5: Add L2 Regularization

# Define L2 regularization factor
k_reg = 0.001

# Create the L2 regularized model
model_l2 = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)), # Input layer = Number of features
    tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(k_reg)),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(k_reg)),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(k_reg)),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the L2 regularized model
model_l2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the L2 regularized model
history_l2 = model_l2.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    verbose=0,
    validation_data=(X_val, y_val)
)

# Obtain final validation accuracy for the L2 regularized model
q5_l2_val_acc = round(float(history_l2.history["val_accuracy"][-1]), 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 5
assert isinstance(model_l2, Sequential), (
    "model_l2 should be a Keras Sequential model."
)
assert isinstance(q5_l2_val_acc, float), (
    "q5_l2_val_acc must be a float."
)
assert 0 <= q5_l2_val_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
# Check that L2 regularization is applied
has_l2 = any(
    layer.kernel_regularizer is not None 
    for layer in model_l2.layers 
    if hasattr(layer, 'kernel_regularizer')
)
assert has_l2, (
    "Model should have L2 regularization. Use kernel_regularizer=l2(0.001)."
)
print(f"L2 Regularized model validation accuracy: {q5_l2_val_acc:.3f}")



# Question 6: Add Dropout

# Create the Dropout model
model_dropout = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)), # Input layer = Number of features
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the Dropout model
model_dropout.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the Dropout model
history_dropout = model_dropout.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    verbose=0,
    validation_data=(X_val, y_val)
)

# Obtain final validation accuracy for the Dropout model
q6_dropout_val_acc = round(float(history_dropout.history["val_accuracy"][-1]), 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 6
assert isinstance(model_dropout, Sequential), (
    "model_dropout should be a Keras Sequential model."
)
assert isinstance(q6_dropout_val_acc, float), (
    "q6_dropout_val_acc must be a float."
)
assert 0 <= q6_dropout_val_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
# Check that Dropout layers are present
has_dropout = any(isinstance(layer, Dropout) for layer in model_dropout.layers)
assert has_dropout, (
    "Model should have Dropout layers. Add Dropout(0.3) after Dense layers."
)
print(f"Dropout model validation accuracy: {q6_dropout_val_acc:.3f}")



# Question 7: Implement Early Stopping

# Create early stopping callback
callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Create a model with early stopping
model_early_stop = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)), # Input layer = Number of features
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model with early stopping
model_early_stop.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model with early stopping
history_early_stopping = model_early_stop.fit(
    X_train,
    y_train,
    epochs=100, # Maximum number of epochs
    batch_size=128,
    verbose=0,
    validation_data=(X_val, y_val),
    callbacks=[callback]
)

# Obtain the epoch at which early stopping occurred
q7_stopped_epoch = history_early_stopping.epoch[-1] + 1
# Obtain final validation accuracy for the early stopping model
q7_early_val_acc = round(float(history_early_stopping.history["val_accuracy"][-1]), 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(model_early_stop, Sequential), (
    "model_early_stop should be a Keras Sequential model."
)
assert isinstance(q7_stopped_epoch, (int, np.integer)), (
    "q7_stopped_epoch must be an integer."
)
assert isinstance(q7_early_val_acc, float), (
    "q7_early_val_acc must be a float."
)
assert q7_stopped_epoch < 100, (
    "Early stopping should have stopped before 100 epochs. "
    "Check your EarlyStopping callback configuration."
)
assert q7_stopped_epoch > 5, (
    "Training stopped too early. Make sure patience=5."
)
print(f"Training stopped at epoch: {q7_stopped_epoch}")
print(f"Early stopping validation accuracy: {q7_early_val_acc:.3f}")



# Question 8: Add Batch Normalization

# Define a model with batch normalization
model_batchnorm = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)), # Input layer = Number of features
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the batch normalization model
model_batchnorm.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the batch normalization model
batchnorm_history = model_batchnorm.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    verbose=0,
    validation_data=(X_val, y_val)
)

# Obtain final validation accuracy for the batch normalization model
q8_batchnorm_val_acc = round(float(batchnorm_history.history["val_accuracy"][-1]), 3)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 8
assert isinstance(model_batchnorm, Sequential), (
    "model_batchnorm should be a Keras Sequential model."
)
assert isinstance(q8_batchnorm_val_acc, float), (
    "q8_batchnorm_val_acc must be a float."
)
assert 0 <= q8_batchnorm_val_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
# Check that BatchNormalization layers are present
has_batchnorm = any(isinstance(layer, BatchNormalization) for layer in model_batchnorm.layers)
assert has_batchnorm, (
    "Model should have BatchNormalization layers."
)
print(f"Batch Normalization model validation accuracy: {q8_batchnorm_val_acc:.3f}")



# Question 9: Compare All Models on Test Set

q9_test_accuracies = {}

# Evaluate all models on the test set
q9_test_accuracies['baseline'] = model_baseline.evaluate(X_test, y_test, verbose=0)[1]
q9_test_accuracies['l2'] = model_l2.evaluate(X_test, y_test, verbose=0)[1]
q9_test_accuracies['dropout'] = model_dropout.evaluate(X_test, y_test, verbose=0)[1]
q9_test_accuracies['early_stopping'] = model_early_stop.evaluate(X_test, y_test, verbose=0)[1]
q9_test_accuracies["batchnorm"] = model_batchnorm.evaluate(X_test, y_test, verbose=0)[1]

# Define the model with the best performance
q9_best_model = max(q9_test_accuracies, key=q9_test_accuracies.get)

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 9
assert isinstance(q9_test_accuracies, dict), (
    "q9_test_accuracies must be a dictionary."
)
expected_keys = {'baseline', 'l2', 'dropout', 'early_stopping', 'batchnorm'}
assert set(q9_test_accuracies.keys()) == expected_keys, (
    f"Dictionary should have keys: {expected_keys}. Got: {set(q9_test_accuracies.keys())}"
)
for key, val in q9_test_accuracies.items():
    assert isinstance(val, float), f"Value for '{key}' must be a float."
    assert 0 <= val <= 1, f"Accuracy for '{key}' must be between 0 and 1."
assert isinstance(q9_best_model, str), (
    "q9_best_model must be a string (the model name)."
)
assert q9_best_model in expected_keys, (
    f"q9_best_model must be one of {expected_keys}."
)
print("Test Accuracies:")
for name, acc in q9_test_accuracies.items():
    print(f"  {name}: {acc:.3f}")
print(f"\nBest model: {q9_best_model}")



# Question 10: Analyze Overfitting Gap

# Function to calculate the overfitting gap using the final epoch
def gap_analysis(history):
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    return round(train_acc - val_acc, 3)

q10_overfit_gaps = {}

# Calculate the overfitting gap for each model
q10_overfit_gaps['baseline'] = gap_analysis(history_baseline)
q10_overfit_gaps['l2'] = gap_analysis(history_l2)
q10_overfit_gaps['dropout'] = gap_analysis(history_dropout)
q10_overfit_gaps['early_stopping'] = gap_analysis(history_early_stopping)
q10_overfit_gaps['batchnorm'] = gap_analysis(batchnorm_history)

# Determine the model with the least overfitting gap
q10_least_overfit = min(q10_overfit_gaps, key=q10_overfit_gaps.get)