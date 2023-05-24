# Decision tree Regression

# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

# Fitting the Decision tree regression model to dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predicting a new result with regression
y_prediction = regressor.predict([[6.5]])
# Output: [150000.]

# Visualizing the decision tree regression result (for higher resolution and smoother curve)
'''
since the Decision tree regression model is non continues we will add the 100 point separated by 0.01 
'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Truth vs Bluff (Decision tree regression model)')
plt.show()  # check the graph folder named 50_position_decision_tree_regression
