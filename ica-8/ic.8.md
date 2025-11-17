ICA8: Building a Text Classification Model using CNN for News Articles
Deadline: Nov 17 at 1:50 pm

Objective:
Build a text classification model that can categorize news articles into one of 20
different categories using a Convolutional Neural Network (CNN). Use the 20 Newsgroups
dataset, which contains a collection of newsgroup posts.

Tasks:

1. Load the 20 Newsgroups dataset, which includes both training and test data.
   You will also need to prepare the text and labels for the training and test sets.

   Here is an example for loading and preparing the training data:thon
   from sklearn.datasets import fetch_20newsgroups

   # Load the 20 Newsgroups dataset (training and test sets)

   train_data = fetch_20newsgroups(subset='train')
   X_train = train_data.data
   y_train = train_data.target
   You need to do the same for the test data.

2. Preprocess the text data by tokenizing the articles and padding the sequences to
   ensure they are all of the same length of 100.

3. Build a Sequential CNN model for text classification using Keras. Here is the
   architecture of the model with the following specific parameters:

   - Embedding Layer:

     - input_dim: 5000 (vocabulary size)
     - output_dim: 100 (embedding dimension)

   - Conv1D Layer:

     - filters: 128
     - kernel_size: 5
     - activation: 'relu'

   - MaxPooling1D Layer:

     - pool_size: 4

   - GlobalMaxPooling1D Layer

   - Dropout Layers:

     - Dropout rate: 0.4

   - Dense Layer:

     - 64 units with 'relu' activation

   - Dropout Layers:

     - Dropout rate: 0.4

   - Dense Layer:

     - 32 units with 'relu' activation

   - Dropout Layers:

     - Dropout rate: 0.4

   - Output Layer:
     - softmax activation (for 20 classes)

4. Compile the model using the adam optimizer and sparse_categorical_crossentropy
   loss function.

5. Train the model on the training data for at least 10 epochs and use the test set for
   validation.

6. Evaluate the model on the test set and report the final loss and accuracy.

Expected Output:
Epoch 1/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 19s 47ms/step - accuracy: 0.0490 - loss: 2.9951 - val_accuracy: 0.1021 - val_loss: 2.9626
Epoch 2/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 20s 46ms/step - accuracy: 0.1198 - loss: 2.8315 - val_accuracy: 0.4241 - val_loss: 1.9559
Epoch 3/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.3576 - loss: 1.9860 - val_accuracy: 0.5433 - val_loss: 1.5151
Epoch 4/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.5073 - loss: 1.5005 - val_accuracy: 0.5959 - val_loss: 1.3351
Epoch 5/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 20s 46ms/step - accuracy: 0.5828 - loss: 1.2262 - val_accuracy: 0.6223 - val_loss: 1.2934
Epoch 6/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 23s 54ms/step - accuracy: 0.6512 - loss: 1.0573 - val_accuracy: 0.6395 - val_loss: 1.2741
Epoch 7/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 18s 46ms/step - accuracy: 0.6969 - loss: 0.9056 - val_accuracy: 0.6488 - val_loss: 1.2889
Epoch 8/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 19s 53ms/step - accuracy: 0.7215 - loss: 0.8163 - val_accuracy: 0.6560 - val_loss: 1.3256
Epoch 9/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 16s 45ms/step - accuracy: 0.7641 - loss: 0.7090 - val_accuracy: 0.6520 - val_loss: 1.3499
Epoch 10/10
354/354 ━━━━━━━━━━━━━━━━━━━━ 16s 46ms/step - accuracy: 0.7837 - loss: 0.6461 - val_accuracy: 0.6589 - val_loss: 1.3997
