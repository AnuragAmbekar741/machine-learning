Linear Regression
Easy
Implement Linear Regression using the closed-form solution (Normal Equation) with NumPy.

You are given:

A feature matrix X of shape (n, d), where n is the number of training samples and d is the number of features
A target vector y of shape (n,)
You must:

Add a bias column (a column of ones) to X
Compute the optimal weight vector w using the pseudo-inverse form of the normal equation:
`w = (X^T X)^† X^T y`

Where † denotes the Moore-Penrose pseudo-inverse. This ensures the solution works even when the matrix is not invertible.

Return the learned weight vector w (including the bias) as a list, rounded to 2 decimal places.

📌 Example input:

`X = [[1], [2], [3]]

y = [2, 4, 6]

`

After solving, you'll return:

`[0.0, 2.0] # Intercept = 0, Slope = 2

`

🧾 Input

X: List of shape (n, d) — input feature matrix
y: List of shape (n,) — target values

📤 Output:

List of shape (d+1,) representing the learned weights, including the bias
Each value must be rounded to 2 decimal places
Return a regular Python list — not a NumPy array

📌 Constraints

1 ≤ n ≤ 100
1 ≤ d ≤ 10
All values in X and y are valid floats
Input may result in a singular matrix — your solution must handle this using the pseudo-inverse
Example 1:

Input: X = [[1], [2], [3]] , y = [2, 4, 6]

Output: [0.0, 2.0]

Explanation:
Binary codes may vary, but decoded result must exactly match 'abac'.

Example 2:

Input: X = [[1, 2], [2, 1], [3, 3]], y = [6, 5, 12]

Output: [-1.0, 1.67, 2.67]
