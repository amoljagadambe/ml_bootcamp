# Natural Language Processing

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
'''
here we are using tab separated file as comma can be found in the review and also we
are ignoring quot in review by adding quoting parameter
'''
dataset = pd.read_csv('data_files/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the reviews
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

'''
we are not using Lemmtization cause of Bag of word model

Download the stopwords from  nltk
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
'''
corpus = []

for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    cleaned_review = ' '.join([ps.stem(word) for word in review if not word in set(all_stopwords)])
    corpus.append(cleaned_review)

# Create the tf-idf model (Iam using tf-idf instead of Bag of words)
from sklearn.feature_extraction.text import TfidfVectorizer

tf_vector = TfidfVectorizer(max_features=500)
X = tf_vector.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# train & test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes classifier to the training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train, y_train)

# predicting the test set results
y_prediction = classifier.predict(x_test)

# Evaluate the model using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

con_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)
print(con_matrix, accuracy_score(y_test, y_prediction))
"""
confusion matrix:
[[56 41] 
 [12 91]] 

Accuracy Matrix 
0.735

"""

# Fitting Random Forest Classification to the training set
from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
forest_classifier.fit(x_train, y_train)

# predicting the test set results
forest_y_prediction = forest_classifier.predict(x_test)

# Evaluate the model using confusion matrix
from sklearn.metrics import confusion_matrix

forest_con_matrix = confusion_matrix(y_true=y_test, y_pred=forest_y_prediction)
print(forest_con_matrix, accuracy_score(y_test, forest_y_prediction))
'''
confusion matrix:
[[84 13]
 [31 72]] 

Accuracy Matrix 
0.78

'''