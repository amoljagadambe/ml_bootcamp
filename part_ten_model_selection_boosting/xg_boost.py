# XGBoost
"""
we don't need feature scaling
"""
# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_files/Data.csv')

# convert the classes
dataset['Class'] = dataset['Class'].replace(to_replace=[2, 4], value=[0, 1])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_prediction = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_prediction)
print(conf_matrix, accuracy_score(y_test, y_prediction))
'''
[[85  2] 
 [ 1 49]] 
Accuracy: 0.9781021897810219
'''

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
'''
Accuracy: 96.53 %
Standard Deviation: 2.63 %
'''