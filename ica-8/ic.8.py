# ========== IMPORTS ==========
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== STEP 1: Load 20 Newsgroups Dataset ==========
# Load training data
train_data = fetch_20newsgroups(subset='train')
X_train = train_data.data
y_train = train_data.target

# Load test data
test_data = fetch_20newsgroups(subset='test')
X_test = test_data.data
y_test = test_data.target

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of classes: {len(set(y_train))}")

# ========== STEP 2: Preprocess Text Data ==========
vocab_size = 5000  # Vocabulary size
max_length = 100   # Sequence length

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to length 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

print(f"Training data shape: {X_train_padded.shape}")
print(f"Test data shape: {X_test_padded.shape}")

# ========== STEP 3: Build CNN Model ==========
model = Sequential([
    # Embedding Layer
    Embedding(input_dim=5000, output_dim=100, input_length=max_length),
    
    # Conv1D Layer
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    
    # MaxPooling1D Layer
    MaxPooling1D(pool_size=4),
    
    # GlobalMaxPooling1D Layer
    GlobalMaxPooling1D(),
    
    # Dropout (0.4)
    Dropout(0.4),
    
    # Dense Layer (64 units, ReLU)
    Dense(64, activation='relu'),
    
    # Dropout (0.4)
    Dropout(0.4),
    
    # Dense Layer (32 units, ReLU)
    Dense(32, activation='relu'),
    
    # Dropout (0.4)
    Dropout(0.4),
    
    # Output Layer (20 classes, softmax)
    Dense(20, activation='softmax')
])

print("Model architecture:")
model.summary()

# ========== STEP 4: Compile Model ==========
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled!")

# ========== STEP 5: Train Model ==========
history = model.fit(
    X_train_padded, y_train,
    epochs=10,
    validation_data=(X_test_padded, y_test),
    verbose=1
)

# ========== STEP 6: Evaluate on Test Set ==========
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"Loss: {test_loss}, Accuracy: {test_accuracy}")