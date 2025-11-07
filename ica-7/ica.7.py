# ========== IMPORTS ==========
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# ========== STEP 1: Load and Normalize CIFAR-10 ==========
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# ========== STEP 2: One-Hot Encode Labels ==========
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Labels converted to one-hot encoding")

# ========== STEP 3: Build CNN Model ==========
model = keras.Sequential([
    # First convolutional block: 32 filters
    layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((3, 3)),
    
    # Second convolutional block: 64 filters
    layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3)),
    
    # Third convolutional block: 128 filters with L2 regularization
    layers.Conv2D(128, (5, 5), padding='same', activation='relu', 
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((3, 3)),
    
    # Flatten layer
    layers.Flatten(),
    
    # Dense layer with 64 neurons
    layers.Dense(64, activation='relu'),
    
    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

print("Model architecture:")
model.summary()

# ========== STEP 4: Compile Model ==========
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled!")

# ========== STEP 5: Train Model ==========
history = model.fit(
    x_train, y_train,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

# ========== STEP 6: Evaluate on Test Set ==========
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy}")