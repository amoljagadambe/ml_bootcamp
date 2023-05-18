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
X[:, 1:3] = si_object.fit_transform(X[:, 1:3])

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
print(x_train)
# feature scaling
'''
here we are scaling the dummy encoded variable also
this will help model to converge faster and we only need to fit
one time like x_train
In classification we don't need to scale y variable

here we have two type 1) standardization 2) normalization
we are using standardization
'''
from sklearn.preprocessing import StandardScaler

standard_scale_x = StandardScaler()
x_train = standard_scale_x.fit_transform(x_train)
x_test = standard_scale_x.transform(x_test)

print(x_train)
