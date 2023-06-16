# Kernel SVM

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

standard_scale_x = StandardScaler()
x_train = standard_scale_x.fit_transform(x_train)
x_test = standard_scale_x.transform(x_test)

# Fitting Kernel SVM to the training set
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
'''
Gaussian Kernel using default gamma this will help us to define circumference of hyperplane
gamma should be non negative
'''
classifier.fit(x_train, y_train)

# predicting the test set results
y_prediction = classifier.predict(x_test)

# Evaluate the model using confusion matrix
from sklearn.metrics import confusion_matrix

con_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)
print(con_matrix)
'''
Output (with rbf kernel): [[64  4] 
                          [ 3 29]]
'''

# Visualising the training set results
from matplotlib.colors import ListedColormap

X_set, y_set = standard_scale_x.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2,
             classifier.predict(standard_scale_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  # check the graphs folder with name kernel_svm_training_set.png
