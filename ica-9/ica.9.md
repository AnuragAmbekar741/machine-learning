ICA9: Generative Deep Learning
Due: Nov 11, 1:50 PM

## Objective

In this assignment, you will use CTGAN (Conditional Tabular GAN) to generate synthetic data
from the Tips dataset. You will implement conditional sampling using one-hot encoded
categorical variables and compare real vs. synthetic distributions.

## Tasks

### 1. Install and import required libraries

- Install CTGAN
- Import required libraries

### 2. Load the dataset

- Load the "tips" dataset from seaborn:

  data = seaborn.load_dataset("tips")

  ### 3. Prepare categorical columns

- Convert the categorical columns to string type

### 4. Prepare conditional columns

- Choose `smoker` and `time` as your conditional columns
- One-hot encode them
- Remove the original conditional columns from the dataset

### 5. Merge encoded and numeric data

- Merge the one-hot encoded conditional data with the rest of the dataset to form your
  final training DataFrame

### 6. Initialize and train a CTGAN model

- Set:
  - `epochs = 1000`
  - `batch_size = 100`
- Treat all original categorical columns except conditional ones as discrete
- Train your model

### 7. Implement a dynamic conditional sampling function

Write a function that:

- Accepts a CTGAN model, a dictionary of one-hot encoded conditions, and `n_samples`
- Forces the generated samples to satisfy the conditions by overwriting the generated
  columns
- Loop over each key-value pair in the conditions dictionary
- For every row in the synthetic DataFrame:
  - Set the corresponding column equal to the value in the dictionary (0 or 1)
  - This ensures the samples obey the one-hot encoded constraints
- Returns the conditioned synthetic samples

The dictionary will look like:n
{
'smoker_Yes': 1,
'smoker_No': 0,
'time_Lunch': 0,
'time_Dinner': 1
}

### 8. Generate conditional samples

- Generate 244 synthetic samples where:
  - The person is a smoker
  - The person is eating dinner
- Print the first few rows

### 9. Generate random (unconditional) samples

- Generate 244 random synthetic samples using the model's standard sampling method
- Print the first few rows

### 10. Compare distributions

**A. Plot histograms**

- Plot the histogram of `total_bill`:
  - Real data
  - Conditional synthetic data

**B. Frequency comparison**

- Print the value counts of:
  - Real `day`
  - Synthetic `day` (from the unconditional samples)

## Expected Output

### Conditional synthetic samples:
