"""
SVR doesn't have feature scaling
"""

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

# Feature scaling
from sklearn.preprocessing import StandardScaler

'''
we need to do this cause SVR doesn't have feature scaling and here
we have use to object for standard scaler
'''
standard_scale_x = StandardScaler()
standard_scale_y = StandardScaler()

# reshape the y to convert it into array
y = y.reshape(-1, 1)  # we need to change this cause standard scaler doesn't accept vector

X = standard_scale_x.fit_transform(X)
y = standard_scale_y.fit_transform(y)

y = y.flatten()  # flattening it again cause regressor throws warning for dependant variable

# Fitting the regression model to dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')  # rbf means gaussian
regressor.fit(X, y)

# predicting a new result with regression
y_prediction = regressor.predict(standard_scale_x.transform([[6.5]]))  # do not fit it again
print(standard_scale_y.inverse_transform([y_prediction]))  # to get the original salary inverse the transform
# Output: [[170370.0204065]]


# Visualizing the SVR result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Truth vs Bluff (SVR model)')
plt.show()  # check the graph folder named 50_position_polynomial_regression
