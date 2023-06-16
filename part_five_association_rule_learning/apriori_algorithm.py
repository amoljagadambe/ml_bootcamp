# Apriori (Rule association learning)
"""
This algorithm works on three base part
    1) Support      (i.e. A user liked M movie)
    2) Confidence   (i.e. A user who like M movie may like N movie)
    3) Lift         (i.e. what are the chances of user who like M movie will like N movie)
"""

# Importing Libraries
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Market_Basket_Optimisation.csv', header=None)

'''
we need list of list in string format for apriori algorithm
'''
dataset = dataset.astype(str)
list_of_transactions = dataset.values.tolist()

# Training Apriori on the dataset
from apyori import apriori

'''
Calculate the support
1) Let's say we consider a product is being bought thrice a day 
    then 3*7/7500 = 0.0028 (bought over week/ total number of transaction)

Calculate the confidence
2) we should always set to less like 20% support on rule

Calculate the lift
3) This gives a insight on association let's put it to 3 as we have only 7500 records

We have to set min length to 2 as it should not associated only single element from transaction
'''
rules = apriori(list_of_transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualizing the results
list_of_results = list(rules)  # rules are already sorted by best relevance


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


result_in_dataframe = pd.DataFrame(inspect(list_of_results),
                                   columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Displaying the results sorted by descending lifts
print(result_in_dataframe.nlargest(n=10, columns='Lift'))
'''
        Left Hand Side Right Hand Side   Support  Confidence      Lift
97                soup            milk  0.003066    0.383333  7.987176
150               soup            milk  0.003066    0.383333  7.987176
96   frozen vegetables            milk  0.003333    0.294118  6.128268
149  frozen vegetables            milk  0.003333    0.294118  6.128268
132  whole wheat pasta       olive oil  0.003866    0.402778  6.128268
59   whole wheat pasta       olive oil  0.003866    0.402778  6.115863
50        tomato sauce       spaghetti  0.003066    0.216981  5.535971
122       tomato sauce       spaghetti  0.003066    0.216981  5.535971
28       fromage blanc           honey  0.003333    0.245098  5.178818
3        fromage blanc           honey  0.003333    0.245098  5.164271

'''
