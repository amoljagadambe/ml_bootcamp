# Eclat
"""
In this algorithm we only consider the support factor
"""
# Importing the libraries
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('data_files/Market_Basket_Optimisation.csv', header=None)
dataset = dataset.astype(str)
list_of_transactions = dataset.values.tolist()

# Training the eclat model on the dataset
from apyori import apriori

rules = apriori(transactions=list_of_transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)

# Displaying the first results coming directly from the output of the apriori function
list_of_results = list(rules)


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


result_in_dataframe = pd.DataFrame(inspect(list_of_results), columns=['Product 1', 'Product 2', 'Support'])

# Displaying the results sorted by descending supports
print(result_in_dataframe.nlargest(n=10, columns='Support'))
'''
              Product 1    Product 2   Support
4         herb & pepper  ground beef  0.015998
7     whole wheat pasta    olive oil  0.007999
2                 pasta     escalope  0.005866
1  mushroom cream sauce     escalope  0.005733
5          tomato sauce  ground beef  0.005333
8                 pasta       shrimp  0.005066
0           light cream      chicken  0.004533
3         fromage blanc        honey  0.003333
6           light cream    olive oil  0.003200

'''