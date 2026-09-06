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

# Question 1: Load and Explore the Dataset

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



# Question 2: Prepare Features and Target

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


# Question 3: Train/Validation/Test Split with Scaling

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

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    verbose=0,
    validation_data=(X_val_scaled, y_val)
)

# Retrieve the training history
history = model.history
# Retrieve the last iteration (-1) and get the accuracy metric
q5_final_train_acc = round(history.history['accuracy'][-1], 3)

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



# Question 6: Plot training curves

# Obtain metrics from the training and validation history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
# Define the range of epochs for plotting
epoch = range(1, len(train_loss) + 1)

# Create figure and axes for the subplots
q6_fig, q6_axes = plt.subplots(nrows=1, ncols=2)

# Plot loss on axis 0 (left subplot)
q6_axes[0].plot(epoch, train_loss, label='Train Loss')
q6_axes[0].plot(epoch, val_loss, label='Validation Loss')
q6_axes[0].set_xlabel('Epoch')
q6_axes[0].set_ylabel('Loss')
q6_axes[0].legend()

# Plot accuracy on axis 1 (right subplot)
q6_axes[1].plot(epoch, train_accuracy, label='Train Accuracy')
q6_axes[1].plot(epoch, val_accuracy, label='Validation Accuracy')
q6_axes[1].set_xlabel('Epoch')
q6_axes[1].set_ylabel('Accuracy')
q6_axes[1].legend()

# Prevent overlapping of subplots
q6_fig.tight_layout()

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 6
assert isinstance(q6_fig, plt.Figure), (
    "q6_fig must be a matplotlib Figure object. "
    "Use fig, axes = plt.subplots(...) and store fig in q6_fig."
)
assert len(q6_fig.axes) == 2, (
    "Figure should have exactly 2 subplots (loss and accuracy)."
)
print("Training curves plotted successfully!")



# Question 7: Evaluate on test set

# Evaluate metrics on the test set
test_loss, test_accuracy = model.evaluate(
    X_test_scaled, 
    y_test,
)

q7_test_metrics = (round(test_loss, 3), round(test_accuracy, 3))

# If all tests pass (there might be hidden tests), you will earn 10 points
# Test Cell: Question 7
assert isinstance(q7_test_metrics, tuple), (
    "q7_test_metrics must be a tuple (loss, accuracy)."
)
assert len(q7_test_metrics) == 2, "Tuple should have exactly 2 elements."
test_loss, test_acc = q7_test_metrics
assert isinstance(test_loss, float) and isinstance(test_acc, float), (
    "Both loss and accuracy must be floats."
)
assert test_loss >= 0, "Loss must be non-negative."
assert 0 <= test_acc <= 1, "Accuracy must be between 0 and 1."
print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")



# Question 8: Manual forward pass function

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def manual_forward_pass(X: np.ndarray, weights: List) -> np.ndarray:
    # Even indices are weights, odd indices are biases
    weight_matrix = weights[0::2]
    bias_matrix = weights[1::2]
    # Calculate the number of hidden layers
    hidden_layers = len(weight_matrix) - 1

    # For hidden layers (ReLU activation)
    for layer in range(hidden_layers):
        W = weight_matrix[layer]
        b = bias_matrix[layer]
        Z = X @ W + b
        A = relu(Z)
        # Set the output of the current layer as input to the next layer
        X = A

    # For output layer (sigmoid activation)
    W = weight_matrix[-1] 
    b = bias_matrix[-1]
    # The input matrix is the output of the last hidden layer
    Z = X @ W + b
    output = sigmoid(Z)

    return np.round(output, 3)

# Get list of weights and bias pair matrices from the model
q8_weights = model.get_weights()
# Perform manual forward pass for the first 5 test samples
q8_manual_preds = manual_forward_pass(X_test_scaled[:5], q8_weights)

# If all tests pass (there might be hidden tests), you will earn 15 points
# Test Cell: Question 8
assert callable(manual_forward_pass), (
    "manual_forward_pass should be a callable function."
)
assert isinstance(q8_manual_preds, np.ndarray), (
    "q8_manual_preds must be a numpy array."
)
assert q8_manual_preds.shape == (5, 1), (
    f"q8_manual_preds should have shape (5, 1), got {q8_manual_preds.shape}. "
    "Make sure you're computing predictions for 5 samples with 1 output."
)
assert np.all((q8_manual_preds >= 0) & (q8_manual_preds <= 1)), (
    "Output probabilities must be between 0 and 1. Did you apply sigmoid to the output?"
)
print(f"Manual forward pass predictions (first 5):\n{q8_manual_preds.flatten()}")



# Question 9: Compare with Logistic Regression Baseline
lr = LogisticRegression(max_iter=1000,random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
q9_logreg_acc = round(accuracy_score(y_test, y_pred), 3)
q9_nn_better = q7_test_metrics[1] > q9_logreg_acc

# If all tests pass (there might be hidden tests), you will earn 15 points
# Test Cell: Question 9
assert isinstance(q9_logreg_acc, float), (
    "q9_logreg_acc must be a float."
)
assert 0 <= q9_logreg_acc <= 1, (
    "Accuracy must be between 0 and 1."
)
assert isinstance(q9_nn_better, bool), (
    "q9_nn_better must be a boolean (True or False)."
)
print(f"Logistic Regression Test Accuracy: {q9_logreg_acc:.3f}")
print(f"Neural Network Test Accuracy: {q7_test_metrics[1]:.3f}")
print(f"Neural Network performed better: {q9_nn_better}")
