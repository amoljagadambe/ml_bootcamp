# Artificial Neural Network

# Importing Libraries
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

label_encoder_x2 = LabelEncoder()
X[:, 2] = label_encoder_x2.fit_transform(X[:, 2])  # for Gender

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X = X[:, 1:]  # Avoiding dummy variable trap

# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
'''
Always apply feature scaling when using neural network
'''
from sklearn.preprocessing import StandardScaler

standard_scale_x = StandardScaler()
x_train = standard_scale_x.fit_transform(x_train)
x_test = standard_scale_x.transform(x_test)

# Importing the keras libraries and package
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# add the layers
'''
selecting units: (1 + number_of_features)/2 = 6
'''

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
'''
> classifier.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 6)                 72        
                                                                 
 dense_1 (Dense)             (None, 6)                 42        
                                                                 
 dense_2 (Dense)             (None, 1)                 7

=================================================================
Total params: 121
Trainable params: 121
Non-trainable params: 0
_________________________________________________________________

'''

# Fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# Predict the result
y_prediction = classifier.predict(x_test)
y_prediction = (y_prediction > 0.5)

# Evaluate the model using confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score

con_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)

'''
> con_matrix
 [[1547   48]
 [ 267  138]]
> accuracy_score(y_test, y_prediction)
0.8425
'''

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

print(classifier.predict(standard_scale_x.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
