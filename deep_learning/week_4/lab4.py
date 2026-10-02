import pathlib
import pickle
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, Dense, Bidirectional, 
    Input, Dropout, LayerNormalization, MultiHeadAttention
)

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

_DATA_PATH = pathlib.Path("imdb_reviews.csv")

if not _DATA_PATH.exists():
    raise FileNotFoundError(
        "imdb_reviews.csv is missing. Please run the download script (w4_download_datasets.py) "
        "or ask the TA for assistance."
    )




# Question 1: Load and Inspect the Dataset
def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the dataset from CSV.
    
    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    # your code here
    return pd.read_csv(path)

df = load_data(_DATA_PATH)
q1_shape: Tuple[int, int] = df.shape
q1_dist: pd.Series = df['label'].value_counts().sort_index()

print(f"Shape: {q1_shape}")
print(f"Class distribution:\n{q1_dist}")
df.head()



# Question 2: Tokenization and Padding
# Global parameters
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 32

# Extract features and labels
X = df.drop(columns=['label']) # Is the same as X = df['text']
y = df['label']

# Create and fit the tokenizer on the training data
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X['text'])

# Convert the test texts to sequences of integers and add padding
X_seq = tokenizer.texts_to_sequences(X['text'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=RANDOM_STATE)



# Question 3: Build a Simple RNN Model
# Create a Simple RNN model
model_rnn = Sequential()
model_rnn.add(Input(shape=(MAX_LEN,)))
model_rnn.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))
model_rnn.add(SimpleRNN(32))
model_rnn.add(Dense(1, activation='sigmoid'))

# Compile the model
model_rnn.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)



# Question 4: Train the Simple RNN Model
history_rnn = model_rnn.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=0
)

# Obtain final validation accuracy
q4_val_acc = round(history_rnn.history['val_accuracy'][-1], 3)



# Question 5: Build an LSTM Model
# Create a LSTM model
model_lstm = Sequential()
model_lstm.add(Input(shape=(MAX_LEN,)))
model_lstm.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile the model
model_lstm.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'],
)



# Question 6: Train the LSTM
history_lstm = model_lstm.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=0
)

# Obtain final validation accuracy
q6_val_acc = round(history_lstm.history['val_accuracy'][-1], 3)



# Question 7: Bidirectional LSTM
# Create a Bidirectional LSTM model
model_bilstm = Sequential()
model_bilstm.add(Input(shape=(MAX_LEN,)))
model_bilstm.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))
model_bilstm.add(Bidirectional(LSTM(32))) # Wrap the LSTM layer with Bidirectional
model_bilstm.add(Dense(1, activation='sigmoid'))

# Compile the model
model_bilstm.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# Train the model
history_bilstm = model_bilstm.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=0
)

# Obtain final validation accuracy
q7_val_acc = round(history_bilstm.history['val_accuracy'][-1], 3)



# Question 8: Transformer Block
from tensorflow.keras.layers import GlobalAveragePooling1D

# Create transformer layers with multi-head attention
inputs = Input(shape=(MAX_LEN,))
embedded = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inputs)
attention = MultiHeadAttention(num_heads=2, key_dim=EMBEDDING_DIM)(embedded, embedded)
pooling = GlobalAveragePooling1D()(attention)
drop1 = Dropout(0.1)(pooling)
dense1 = Dense(20, activation='relu')(drop1)
drop2 = Dropout(0.1)(dense1)
dense2 = Dense(1, activation='sigmoid')(drop2)

# Create the transformer model
model_transformer = Model(inputs=inputs, outputs=dense2)

# Compile the model
model_transformer.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model_transformer.summary()



# Question 9: Train Transformer
history_transformer = model_transformer.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=0
)

# Obtain final validation accuracy
q9_val_acc = round(history_transformer.history['val_accuracy'][-1], 3)



# Question 10: Model Comparison
q10_results = {
    'SimpleRNN': q4_val_acc,
    'LSTM': q6_val_acc,
    'BiLSTM': q7_val_acc,
    'Transformer': q9_val_acc
}

q10_best_model = max(q10_results, key=q10_results.get)