# Regression Template

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

'''# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

# Fitting the regression model to dataset
regressor = None

# predicting a new result with regression
y_prediction = regressor.predict([[6.5]])


# Visualizing the regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Truth vs Bluff (regression model)')
plt.show()  # check the graph folder named 50_position_polynomial_regression


