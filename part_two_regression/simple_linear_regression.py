# Simple linear regression

# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting simple linear regression model to training set
from sklearn.linear_model import LinearRegression
'''
this model will automatically scale the independent variable
'''
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_predicted = regressor.predict(x_test)

# visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# visualizing the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
