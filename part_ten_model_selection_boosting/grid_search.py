# Grid Search i.e.Hyperparameter tuning/ choosing the model

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_files/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_prediction = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_prediction)
print(conf_matrix, accuracy_score(y_test, y_prediction))
'''
[[64  4] 
 [ 3 29]] 
Accuracy: 0.93

'''

# Applying grid search to find best model and best parameters
from sklearn.model_selection import GridSearchCV

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}
]

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
'''
Best Accuracy: 90.67 %
Best Parameters: {'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'}

'''