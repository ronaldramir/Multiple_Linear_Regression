# Multiple_Linear_Regression

## Simple Linear Regression with Gradient Descent in Python
This code implements the Gradient Descent algorithm from scratch to perform linear regression on the Boston housing dataset. It compares the results with the LinearRegression model from scikit-learn.

## Dependencies
numpy
pandas
matplotlib.pyplot
scikit-learn
Instructions
Install dependencies:

## Bash
pip install numpy pandas matplotlib scikit-learn
Usa el código con precaución.

## Code Explanation
The code performs the following steps:

Imports necessary libraries (numpy, pandas, matplotlib, scikit-learn).
Loads the Boston housing dataset using pandas.
Defines features (x_train) by dropping the target variable (medv) from the DataFrame.
Defines the target variable (y_train) as the medv column.
Normalizes features using StandardScaler from scikit-learn.
Converts data to NumPy arrays for efficiency in calculations.
Defines functions:
compute_cost: Calculates the mean squared error cost function.
compute_gradient: Calculates the gradient of the cost function with respect to the weights and intercept.
gradient_descent: Implements the gradient descent algorithm to iteratively update weights and intercept to minimize the cost function.
Initializes parameters:
Weights (w_init) are set to zeros.
Intercept (b_init) is set to zero.
Learning rate (alpha) is adjusted for optimal performance.
Number of iterations (iterations) is adjusted for convergence.
Runs gradient descent using the defined function.
Prints the optimized weights and intercept obtained from gradient descent.
Plots the cost history to visualize the convergence of the algorithm.
Compares with scikit-learn:
Creates a LinearRegression model and fits it to the data.
Prints the weights, intercept, and R-squared score from scikit-learn.
Calculates predictions:
Uses the trained scikit-learn model to predict values.
Uses the final weights and intercept from gradient descent to predict values.
Plots predictions vs. true values:
Creates two subplots to visualize predictions from scikit-learn and gradient descent compared to the true values.
Includes an ideal fit line for reference.
Output

## The code outputs the following:

Optimized weights and intercept from gradient descent.
Cost history plot showing convergence.
Weights, intercept, and R-squared score from scikit-learn.
Two scatter plots:
Scikit-learn predictions vs. true values.
Gradient descent predictions vs. true values.
Both plots include an ideal fit line for comparison.
This allows you to compare the performance of your custom gradient descent implementation with scikit-learn's optimized linear regression model.
