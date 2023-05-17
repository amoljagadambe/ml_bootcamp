# Data preprocessing

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer

si_object = SimpleImputer(missing_values=np.nan, strategy='mean')
si_object = si_object.fit(X[:, 1:3])
X[:, 1:3] = si_object.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder_country = LabelEncoder()
X[:, 0] = label_encoder_country.fit_transform(X[:, 0])
'''
above line will create class like 0,1,2 in ML
the algorithm will co relate on basis of the class 
value means 0 is less than 1 ans 1 is less than 2
so we need dummy encoding like one hot encoding
'''

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

'''
dependant variable will have no correlation
'''
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# train & test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
