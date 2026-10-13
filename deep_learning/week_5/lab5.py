import pathlib
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Layer, Embedding, Add, GlobalAveragePooling1D, Flatten, Reshape
)

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("fashion_mnist_vit.csv")

if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "fashion_mnist_vit.csv is missing. Please run the download script (w5_download_datasets.py) "
        "or ask the TA for assistance."
    )

# Class names for Fashion MNIST
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Question 1: Load and Prepare Image Data
def load_and_prepare_data(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    # Split the data into training and test sets
    X_train = df.loc[df["split"].eq("train"), ~df.columns.isin(["label", "split"])].to_numpy()
    X_test = df.loc[df["split"].eq("test"), ~df.columns.isin(["label", "split"])].to_numpy()

    y_train = df.loc[df["split"].eq("train"), "label"].to_numpy()
    y_test = df.loc[df["split"].eq("test"), "label"].to_numpy()
    
    # Reshape the image data to 28x28 pixels and normalize it to [0, 1]
    X_train = X_train.reshape(-1, 28, 28)/255.0
    X_test = X_test.reshape(-1, 28, 28)/255.0
 
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prepare_data(_DATA_PATH)
q1_shape: Tuple[int, int, int] = X_train.shape

print(f"Training data shape: {q1_shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Visualize some samples
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(CLASS_NAMES[y_train[i]])
    ax.axis('off')
plt.suptitle("Sample Images from Fashion MNIST")
plt.tight_layout()
plt.show()



# Question 2: Positional Encoding
def create_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Create sinusoidal positional encoding.
    
    Parameters
    ----------
    seq_len : int
        Sequence length (number of positions).
    d_model : int
        Embedding dimension.
        
    Returns
    -------
    np.ndarray
        Positional encoding of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            # Use floor division to keep same frequency for sin and cos pairs
            arg = pos / np.power(10000, 2 * (i // 2) / d_model)
            if i % 2 == 0: # Even index: use sine
                pe[pos, i] = np.sin(arg)
            else: # Odd index: use cosine
                pe[pos, i] = np.cos(arg)
    return pe

# For ViT with 7x7 patches on 28x28 images: 49 patches
SEQ_LEN = 49
D_MODEL = 64

q2_pos_enc: np.ndarray = create_positional_encoding(SEQ_LEN, D_MODEL)

print(f"Positional encoding shape: {q2_pos_enc.shape}")

# Visualize the positional encoding
plt.figure(figsize=(10, 4))
plt.imshow(q2_pos_enc, aspect='auto', cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar()
plt.show()



# Question 3: Scaled Dot-Product Attention
from scipy.special import softmax

def scaled_dot_product_attention(
    query: np.ndarray, 
    key: np.ndarray, 
    value: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.
    
    Parameters
    ----------
    query : np.ndarray
        Query matrix of shape (seq_len, d_k).
    key : np.ndarray
        Key matrix of shape (seq_len, d_k).
    value : np.ndarray
        Value matrix of shape (seq_len, d_v).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (output, attention_weights)
        - output: shape (seq_len, d_v)
        - attention_weights: shape (seq_len, seq_len)
    """
    d_k = query.shape[1] # Index 1 is dimension of the key/query vectors
    attention = softmax(query @ key.T / np.sqrt(d_k), axis=1) # Apply softmax per query/row
    output = attention @ value
    return output, attention

# Test with simple example
np.random.seed(RANDOM_STATE)
test_seq_len = 4
test_d_k = 8

Q = np.random.randn(test_seq_len, test_d_k).astype(np.float32)
K = np.random.randn(test_seq_len, test_d_k).astype(np.float32)
V = np.random.randn(test_seq_len, test_d_k).astype(np.float32)

q3_output, q3_attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"Attention weights shape: {q3_attention_weights.shape}")
print(f"Output shape: {q3_output.shape}")
print(f"Attention weights (rows sum to 1):\n{q3_attention_weights}")



# Question 4: Layer Normalization
def layer_norm(
    x: np.ndarray, 
    gamma: np.ndarray, 
    beta: np.ndarray, 
    epsilon: float = 1e-6
) -> np.ndarray:
    """Apply layer normalization.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of shape (..., features).
    gamma : np.ndarray
        Scale parameter of shape (features,).
    beta : np.ndarray
        Shift parameter of shape (features,).
    epsilon : float
        Small constant for numerical stability.
        
    Returns
    -------
    np.ndarray
        Normalized array, same shape as input.
    """
    # Compute mean and variance along the rows for layer normalization, not batch normalization
    # keepdims is to mantain the original dimension for future operations
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    output = (gamma*(x - mean)/np.sqrt(variance + epsilon)) + beta
    return output

# Test layer normalization
np.random.seed(RANDOM_STATE)
test_input = np.random.randn(3, 8).astype(np.float32) * 10 + 5  # Mean ~5, varied values
gamma = np.ones(8, dtype=np.float32)
beta = np.zeros(8, dtype=np.float32)

q4_normalized: np.ndarray = layer_norm(test_input, gamma, beta)

print(f"Input mean per sample: {test_input.mean(axis=-1)}")
print(f"Normalized mean per sample: {q4_normalized.mean(axis=-1)}")
print(f"Normalized std per sample: {q4_normalized.std(axis=-1)}")



# Question 5: Patch Embedding for Vision Transformer
def extract_patches(images: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches from images.
    
    Parameters
    ----------
    images : np.ndarray
        Images of shape (batch_size, height, width).
    patch_size : int
        Size of each square patch.
        
    Returns
    -------
    np.ndarray
        Patches of shape (batch_size, num_patches, patch_dim).
        patch_dim = patch_size * patch_size
    """
    # Extract image shape
    batch_size, height, width = images.shape
    # Compute the number of vertical and horizontal patches
    # Floor division to avoid overlapping patches
    vertical_patches = height // patch_size
    horizontal_patches = width // patch_size
    # Calculate total number of patches and patch dimension
    num_patches = vertical_patches * horizontal_patches
    patch_dim = patch_size * patch_size
    # Define output array, initialized with zeros for now
    patches = np.zeros((batch_size, num_patches, patch_dim))
    # Current patch index
    index = 0 

    for i in range(0, height, patch_size): # Iteration over vertical patches
        for j in range(0, width, patch_size): # Iteration over horizontal patches
            # Extract the current patch for all images in the batch
            batch_patch = images[:, i:i + patch_size, j:j + patch_size]
            # Flatten the current patch (for all images in the batch) into a vector
            patches[:, index, :] = batch_patch.reshape(batch_size, patch_dim)
            # Move to the next patch
            index += 1
    
    return patches

PATCH_SIZE = 4
q5_patches: np.ndarray = extract_patches(X_train[:100], PATCH_SIZE)

print(f"Patches shape: {q5_patches.shape}")
print(f"Expected: (100, 49, 16)")

# Visualize patches from first image
fig, axes = plt.subplots(1, 7, figsize=(14, 2))
fig.suptitle("First 7 patches from first image (each 4x4)")
for i in range(7):
    patch = q5_patches[0, i].reshape(PATCH_SIZE, PATCH_SIZE)
    axes[i].imshow(patch, cmap='gray')
    axes[i].set_title(f"Patch {i}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()



# Question 6: Build a Simple Vision Transformer
# Model parameters
NUM_PATCHES = 49  # 7x7 patches
PATCH_DIM = 16    # 4x4 flattened
EMBED_DIM = 64
NUM_HEADS = 4
KEY_DIM = 16
FF_DIM = 128
NUM_CLASSES = 10

# Create vision transformer layer with multi-head self-attention
# Input: Flattened patches [num_patches, patch_dim]
input = Input(shape=(NUM_PATCHES, PATCH_DIM))
# Use a dense layer to project patches into the embedding linear space
linear_projection = Dense(EMBED_DIM)(input)
# Add positional encoding
pos_encoder = Dense(q2_pos_enc.shape[-1])(linear_projection) 
# Apply multi-head self-attention on the positionally encoded patches
attention = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(pos_encoder, pos_encoder)
# Apply layer normalization to the attention output + residual connection (positionally encoded input)
attention_norm = LayerNormalization()(attention + pos_encoder)
# Feed-forward networks
ffn1 = Dense(FF_DIM, activation='relu')(attention_norm)
ffn2 = Dense(EMBED_DIM)(ffn1) # Project back to embedding dimension
# Apply layer normalization to the FFN output + residual connection (normalized attention output)
ffn_norm = LayerNormalization()(ffn2 + attention_norm)
# Global Average Pooling
global_avg_pool = GlobalAveragePooling1D()(ffn_norm)
# Dense layer
dense2 = Dense(EMBED_DIM, activation='relu')(global_avg_pool)
# Add dropout
dropout = Dropout(0.3)(dense2) 
# Output layer for classification
output = Dense(NUM_CLASSES, activation='softmax')(dropout)

# Build the VT model
model_vit = Model(inputs=input, outputs=output)

model_vit.summary()



# Question 7: Train the Vision Transformer
# Extract patches from training and test images
X_train_patches = extract_patches(X_train, PATCH_SIZE)
X_test_patches = extract_patches(X_test, PATCH_SIZE)

# Compile the Vision Transformer model
model_vit.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Vision Transformer on the extracted patches
history_vit = model_vit.fit(
    X_train_patches,
    y_train,
    validation_data=(X_test_patches, y_test),
    epochs=3,
    batch_size=64,
    verbose=0
)

q7_val_acc = round(history_vit.history['val_accuracy'][-1], 3)



# Question 8: Compare with Simple CNN
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# Define a simple CNN model
model_cnn = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the CNN model
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the CNN model
history_cnn = model_cnn.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=64,
    verbose=0
)

q8_val_acc = round(history_cnn.history['val_accuracy'][-1], 3)



# Question 9: Understanding BERT vs GPT architecture
q9_bert_direction = 'bidirectional'
q9_gpt_direction = 'causal'
q9_bert_task = 'understanding'
q9_gpt_task = 'generation'

print(f"BERT attention: {q9_bert_direction}")
print(f"GPT attention: {q9_gpt_direction}")
print(f"BERT best for: {q9_bert_task}")
print(f"GPT best for: {q9_gpt_task}")



# Question 10: Model Comparison and Analysis
q10_results = {
    'ViT': q7_val_acc,
    'CNN': q8_val_acc
}

q10_best_model = max(q10_results, key=q10_results.get)
q10_vit_params = model_vit.count_params()
q10_cnn_params = model_cnn.count_params()