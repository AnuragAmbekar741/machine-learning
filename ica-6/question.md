ICA 6: Time Series Forecasting
Deadline: 10/31/2025 @ 1:50 pm
Objective: In this assignment, you will implement a Gated Recurrent Unit (GRU) model to
forecast stock prices based on multiple features.
Instructions:

1. Load the Tesla stock dataset from a CSV file (tesla_stock_data.csv).
2. Normalize the features: Open, High, Low, Volume, and Close using Min-Max scaling.
3. Create a dataset suitable for time series forecasting. Use the past 30 days' data to predict the closing price.
4. Split the dataset into training (80%) and testing (20%) sets.
5. Build a GRU model using Keras with the following specifications:
   a. Input shape should correspond to the feature set.
   b. Include a single GRU layer with 80 units
   c. Include a dense output layer to predict the closing price.
6. Compile the model with the Adam optimizer and mean squared error loss function.
7. Train the model for 50 epochs with a batch size of 32.
8. After training, use the model to make predictions on the test set.
9. Evaluate the model's performance using Root Mean Squared Error (RMSE) and Mean
   Absolute Error (MAE).
   Expected Output:
   Epoch 1/50
   50/50 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - loss: 0.0191
   Epoch 2/50
   50/50 ━━━━━━━━━━━━━━━━━━━━ 1s 18ms/step - loss: 2.5334e-04
   .
   .
   .
   Epoch 48/50
   50/50 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 9.1858e-05
   Epoch 49/50
   50/50 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 8.9993e-05
   Epoch 50/50
   50/50 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - loss: 1.0017e-04
   13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
   Root Mean Squared Error (RMSE): 0.03326090470231709
   Mean Absolute Error (MAE): 0.025061157442621714
