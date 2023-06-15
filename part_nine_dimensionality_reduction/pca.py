# PCA (Principal component analysis)

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

standard_scale_x = StandardScaler()
x_train = standard_scale_x.fit_transform(x_train)
x_test = standard_scale_x.transform(x_test)

# Apply PCA (reduce dimension)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # we replaced the value 2 after getting below ratio
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)
'''
> pca.explained_variance_ratio_

[0.35952175 0.19820577 0.14223855 0.11215211 0.06056091 0.03559827
 0.02705868 0.01838811 0.01375635 0.01007092 0.00979244 0.00798524
 0.0046709 ]

here first two principal component explained 57% variance, we are only
taking these two because of the output graphs.
Now let's take only 2 components
'''
# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predicting the test set results
y_prediction = classifier.predict(x_test)

# Evaluate the model using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

con_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)
print(con_matrix, accuracy_score(y_test, y_prediction))
'''
Output: 
 [[ 0  1 13]                    
  [ 4 11  1]                    
  [ 6  0  0]]
'''

# Visualising the training set results
from matplotlib.colors import ListedColormap

X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2,
             classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()  # check the graphs folder with name Logistic_regression_using_pca.png
