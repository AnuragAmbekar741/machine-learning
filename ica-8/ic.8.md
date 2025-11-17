ICA8: Building a Text Classification Model using CNN for News Articles
Deadline: Nov 17 at 1:50 pm
Objective: Build a text classification model that can categorize news articles into one of 20
different categories using a Convolutional Neural Network (CNN). Use the 20 Newsgroups
dataset, which contains a collection of newsgroup posts.
Tasks:

First, you will load the 20 Newsgroups dataset, which includes both training and test
data. You will also need to prepare the text and labels for the training and test sets.
Here is an example for loading and preparing the training data:
from sklearn.datasets import fetch_20newsgroups

# Load the 20 Newsgroups dataset (training and test sets) train_data

= fetch_20newsgroups(subset='train')
X_train = train_data.data
y_train = train_data.target 2. 3. 4. you need to do the same for the test data.
Preprocess the text data by tokenizing the articles and padding the sequences to
ensure they are all of the same length of 100.
Build a Sequential CNN model for text classification using Keras. Here is the
architecture of the model with the following specific parameters:
Embedding Layer:
input_dim: 5000 (vocabulary size)
output_dim: 100 (embedding dimension)
Conv1D Layer:
filters: 128
kernel_size: 5
activation: 'relu'
MaxPooling1D Layer:
pool_size: 4
GlobalMaxPooling1D Layer
Dropout Layers:
Dropout rate: 0.4
Dense Layer:
64 units with 'relu'activation.
Dropout Layers: Dropout
rate: 0.4
Dense Layer:
32 units with 'relu'activation. 5. 6. 7. Dropout Layers:
Dropoutrate: 0.4
Output Layer:
softmax activation (for 20 classes)
Compile the model using the adam optimizer and sparse_categorical_crossentropy
loss function.
Train the model on the training data for at least 10 epochs and use the test set for
validation.
Evaluate the model on the test set and report the final loss and accuracy.
Expected Output:
Epoch 1/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 20s 51ms/step - accuracy: 0.0519 - loss: 2.9929 - val_accuracy: 0.1474 - val_loss: 2.8766
Epoch 2/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 19s 53ms/step - accuracy: 0.1500 - loss: 2.6748 - val_accuracy: 0.4263 - val_loss: 1.9375
Epoch 3/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 17s 48ms/step - accuracy: 0.3530 - loss: 1.9389 - val_accuracy: 0.5552 - val_loss: 1.5433
Epoch 4/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 23s 55ms/step - accuracy: 0.5025 - loss: 1.4874 - val_accuracy: 0.6002 - val_loss: 1.3722
Epoch 5/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 18s 47ms/step - accuracy: 0.5946 - loss: 1.2152 - val_accuracy: 0.6273 - val_loss: 1.2777
Epoch 6/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.6631 - loss: 1.0046 - val_accuracy: 0.6365 - val_loss: 1.2981
Epoch 7/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 22s 50ms/step - accuracy: 0.7034 - loss: 0.8761 - val_accuracy: 0.6374 - val_loss: 1.3587
Epoch 8/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.7441 - loss: 0.7713 - val_accuracy: 0.6490 - val_loss: 1.3760
Epoch 9/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.7781 - loss: 0.6692 - val_accuracy: 0.6523 - val_loss: 1.4219
Epoch 10/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 17s 47ms/step - accuracy: 0.7960 - loss: 0.5964 - val_accuracy: 0.6553 - val_loss: 1.4728
236/236 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.6692 - loss: 1.4503 Loss:
1.4728387594223022, Accuracy: 0.6553372144699097
