# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset


# Fitting Polynomial Regression to the dataset


# Visualising the Linear Regression results

# Visualising the Polynomial Regression results


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# Predicting a new result with Linear Regression


# Predicting a new result with Polynomial Regression
