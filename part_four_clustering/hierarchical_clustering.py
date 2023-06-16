# Hierarchical clustering

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values  # we don't have dependant variable in clustering

# Using dendrogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))  # we are minimizing the variance here using ward method
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance ')
plt.show()  # check the graphs folder with name optimal_cluster_using_dendrogram_method_hc.png

# Fitting hierarchical clustering to mall dataset
'''
There are two type of hierarchical clustering
    1) Agglomerated (consider each point as cluster)
    2) Divisive  

we will use the first one in this example
'''
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()  # check the graphs folder with name number_of_clusters_in_hierarchical_clustering.png

