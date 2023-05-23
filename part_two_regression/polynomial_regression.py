# Polynomial Regression

# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Position_Salaries.csv')
'''
We are ignoring the position or title as till will have no impact as much as level
and we converted a single vector to array for X variable
'''
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the linear regression to dataset
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting the polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures

poly_regressor = PolynomialFeatures(degree=3)  # change the degree to add more poly variable
X_ploy = poly_regressor.fit_transform(X)  # converted the X to polynomial variables y = b0 + b1x1 + b2x1^2

linear_poly_regression = LinearRegression()
linear_poly_regression.fit(X_ploy, y)

# # Visualizing the linear regression result
# plt.scatter(X, y, color='red')
# plt.plot(X, linear_regressor.predict(X), color='blue')
# plt.xlabel('Position Level')
# plt.ylabel('Salaries')
# plt.title('Truth vs Bluff (linear regression)')
# plt.show()  # check the graph folder named 50_position_linear_regression
#
# # Visualizing the polynomial regression result
# plt.scatter(X, y, color='red')
# plt.plot(X, linear_poly_regression.predict(X_ploy), color='blue')
# plt.xlabel('Position Level')
# plt.ylabel('Salaries')
# plt.title('Truth vs Bluff (Polynomial regression)')
# plt.show()  # check the graph folder named 50_position_polynomial_regression

# predicting a new result with linear regression
linear_prediction = linear_regressor.predict([[6.5]])  # use array not vector
# Output: [330378.78787879]

# predicting a new result with linear regression
polynomial_prediction = linear_poly_regression.predict(poly_regressor.fit_transform([[6.5]]))  # transform the array
# Output: [133259.46969697]
