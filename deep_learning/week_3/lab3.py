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



# Question 3: Build a Simple CNN
# Create the CNN model
model_cnn = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(32, (3, 3), activation='relu', padding='same'), # 32 filters (3x3 kernel size)
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
    Conv2D(64, (3, 3), activation='relu', padding='same'), # 64 filters (3x3 kernel size)
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
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



# Question 4: Calculate Output Dimensions

l1 = (32, 32, 32) # After first convolution layer
l2 = (16, 16, 32) # After first MaxPooling layer
l3 = (16, 16, 64) # After second convolution layer
l4 = (8, 8, 64) # After second MaxPooling layer

q4_output_shapes = (l1, l2, l3, l4)



# Question 5: Train the CNN
# Train the model
history_cnn = model_cnn.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=0
)

# Get the final validation accuracy
q5_val_acc = round(history_cnn.history['val_accuracy'][-1], 3)



# Question 6: Add Data Augmentation
# Create ImageDataGenerator object
datagen = ImageDataGenerator(
    rotation_range=15, # Rotate up to 15 degrees
    width_shift_range=0.1, # Shift horizontally up to 10%
    height_shift_range=0.1, # Shift vertically up to 10%
    horizontal_flip=True # Randomly flip horizontally
)

# New CNN with the same architecture as model_cnn
model_aug = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(32, (3, 3), activation='relu', padding='same'), # 32 filters (3x3 kernel size)
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
    Conv2D(64, (3, 3), activation='relu', padding='same'), # 64 filters (3x3 kernel size)
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the CNN model with augmented data
model_aug.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with augmented data
history_aug = model_aug.fit(
    datagen.flow(X_train, y_train, batch_size=64), # Use data augmentation for training
    epochs=5,
    validation_data=(X_val, y_val),
    verbose=0
)

# Get the final validation accuracy for the augmented model
q6_aug_val_acc = round(history_aug.history['val_accuracy'][-1], 3)




# Question 7: Build a Deeper CNN with Batch Normalization
# Create a deeper CNN model with batch normalization
model_deep = Sequential([
    Input(shape=X_train.shape[1:]),

    # BLock 1
    Conv2D(32, (3, 3), activation='relu', padding='same'), # 32 filters (3x3 kernel size)
    BatchNormalization(),
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
    Dropout(0.25), # Dropout of 25%

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'), # 64 filters (3x3 kernel size)
    BatchNormalization(),
    MaxPooling2D((2, 2)), # Max pooling of 2x2 size
    Dropout(0.25), # Dropout of 25%

    # Classification head
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), # Dropout of 50%
    Dense(10, activation='softmax')
])

# Compile the deeper CNN model
model_deep.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the deeper CNN model
history_deep = model_deep.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_data=(X_val, y_val),
    verbose=0
)

# Calculate number of parameters
q7_deep_params = model_deep.count_params()

# Get the final validation accuracy for the deeper model
q7_deep_val_acc = round(history_deep.history['val_accuracy'][-1], 3)




# Question 8: Confusion Matrix Analysis
# Make predictions on test set using the deep CNN model
y_pred = model_deep.predict(X_test, verbose=0)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
q8_conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Create a copy of the confusion matrix
conf_matrix_copy = q8_conf_matrix.copy()

# Set the diagonal values to zero to ignore correct predictions
np.fill_diagonal(conf_matrix_copy, 0)

# Get the flat index that correspond to the highest off-diagonal value
most_confused_indices = np.argmax(conf_matrix_copy)

# Converts the flat index to a tuple of row and column indices
q8_most_confused = np.unravel_index(most_confused_indices, conf_matrix_copy.shape)



# Question 9: Compare All Models on Test Set
q9_test_accuracies = {}

# Evaluate test set for all models
# evaluate() return [loss, accuracy]
q9_test_accuracies['simple_cnn'] = round(model_cnn.evaluate(X_test, y_test, verbose=0)[1], 3)
q9_test_accuracies['augmented'] = round(model_aug.evaluate(X_test, y_test, verbose=0)[1], 3)
q9_test_accuracies['deep'] = round(model_deep.evaluate(X_test, y_test, verbose=0)[1], 3)

# Model with highest accuracy
q9_best_model = max(q9_test_accuracies, key=q9_test_accuracies.get)



# Question 10: Visualize Learned Filters
# Get the weights of the first convolutional layer
model_cnn_weights = model_cnn.layers[0].weights[0]
# Get the shape of the filters in the first convolutional layer
q10_filter_shape = tuple(model_cnn_weights.shape)
