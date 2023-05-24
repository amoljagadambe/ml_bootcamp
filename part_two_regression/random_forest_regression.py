# Random forest Regression

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

# Fitting the Random forest regression model to dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)  # tweak the estimator to get accurate result
regressor.fit(X, y)

# predicting a new result with regression
y_prediction = regressor.predict([[6.5]])
# Output: [167000.] with 10 tree
# Output: [160333.33333333] with 300 tree

# Visualizing the Random forest regression result (for higher resolution and smoother curve)
'''
since the Random forest regression model is non continues we will add the 100 point separated by 0.01 
'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Truth vs Bluff (Random forest regression model)')
plt.show()  # check the graph folder named 50_position_random_forest_regression
