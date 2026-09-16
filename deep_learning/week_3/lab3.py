import pathlib
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    BatchNormalization, Dropout, GlobalAveragePooling2D, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings for cleaner output
tf.get_logger().setLevel("ERROR")

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# CIFAR-10 class names for reference
CLASS_NAMES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Image dimensions
IMG_HEIGHT: int = 32
IMG_WIDTH: int = 32
IMG_CHANNELS: int = 3
NUM_CLASSES: int = 10

_TRAIN_PATH = pathlib.Path("cifar10_train.csv")
_TEST_PATH = pathlib.Path("cifar10_test.csv")

if not _TRAIN_PATH.exists() or not _TEST_PATH.exists():
    raise FileNotFoundError(
        "CIFAR-10 CSV files are missing from the lab directory. "
        "Please run the download script (w3_download_datasets.py) or ask the TA for assistance."
    )



# Question 1: Load and Explore the Dataset
def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load a CIFAR-10 CSV file.

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



# Question 2: Preprocess and Reshape Data
# Load test data as well
test_df = load_data(_TEST_PATH)

# Normalize pixel values to the range [0, 1]
X = train_df.drop(columns="label").to_numpy()/255
y = train_df["label"]

# Reshaping flat pixels to image format
X = X.reshape(-1, 32, 32, 3)

# Split the data into training and validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    train_size=0.75,
    random_state=RANDOM_STATE,
)

# Calculate set size
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
assert len(X_train.shape) == 4, (
    "X_train should be 4D: (samples, height, width, channels). "
    "Use reshape(-1, 32, 32, 3)."
)
assert X_train.shape[1:] == (32, 32, 3), (
    f"Each image should be (32, 32, 3), got {X_train.shape[1:]}"
)
print(f"Train/Val/Test split: {n_train} / {n_val} / {n_test}")
print(f"Image shape: {X_train.shape[1:]}")



# Question 3: Build a Simple CNN
# Create the CNN model
model_cnn = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(
        32, # Number of filters
        (3, 3), # Kernel size
        activation='relu', 
        padding='same'
    ),
    MaxPooling2D((2, 2)), # Max pooling
    Conv2D(
        64, # Number of filters
        (3, 3), # Kernel size
        activation='relu',
        padding='same'
    ),
    MaxPooling2D((2, 2)),
    Conv2D(
        128, # Number of filters
        (3, 3), # Kernel size
        activation='relu',
        padding='same'
    ),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the CNN model
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

q3_n_params = model_cnn.count_params()