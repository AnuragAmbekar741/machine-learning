ICA7: Convolutional Neural Network (CNN)
Due 11/7/2025 at 1:50 pm.
Objective:
In this assignment, you will build a Convolutional Neural Network (CNN) for image
classification on the CIFAR-10 dataset. You will also implement L2 regularization to help
prevent overfitting and use the SGD optimizer for training. By the end of the assignment,
you will be able to evaluate your model's performance on a test set.
Tasks:

1. load the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10
   classes. Normalize the pixel values to the range [0, 1].
2. Convert the labels into one-hot encoded format.
3. Use Keras' Sequential API to define the CNN model architecture. The architecture
   should consist of:
   a. Three convolutional blocks with increasing number of filters (32, 64, 128).
   Each filter in the convolutional layer will have a 5x5 size. The third block
   should have an L2 regularization of 0.01 applied to its convolutional layer.
   b. Use padding='same'
   c. Three Max-pooling layers with 3x3 kernels.
   d. A flattening layer to flatten the 3D output to 1D.
   e. A dense layer with 64 neurons and ReLU activation.
   f. An output layer.
4. Compile the model using the Stochastic Gradient Descent (SGD) optimizer with a
   learning rate of 0.1. Use an appropriate loss function and track accuracy as a
   metric.
5. Train the model for 3 epochs. Use the validation_split parameter to hold out 20% of
   the training data for validation during training.
6. After training, evaluate the model on the test dataset and print the test accuracy.
   Note: Running only 3 iterations is not recommended for actual training or achieving
   good accuracy; it is used here solely to save time due to class time limits.
   Expected Output:
   Epoch 1/3
   625/625 ━━━━━━━━━━━━━━━━━━━━ 105s 166ms/step - accuracy:
   0.1973 - loss: 2.6816 - val_accuracy: 0.4372 - val_loss: 1.7115
   Epoch 2/3
   625/625 ━━━━━━━━━━━━━━━━━━━━ 100s 161ms/step - accuracy:
   0.4308 - loss: 1.6970 - val_accuracy: 0.5168 - val_loss: 1.4314
   Epoch 3/3
   625/625 ━━━━━━━━━━━━━━━━━━━━ 95s 153ms/step - accuracy:
   0.5091 - loss: 1.4611 - val_accuracy: 0.5594 - val_loss: 1.3508
   313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - accuracy: 0.5535 -
   loss: 1.3442
   Test accuracy: 0.5475999712944031
