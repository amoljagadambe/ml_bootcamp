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
results = list(rules)  # rules are already sorted by best relevance

"""
Top 5 association rule

[RelationRecord(items=frozenset({'chicken', 'light cream'}), support=0.004532728969470737, ordered_statistics=[
OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), 
confidence=0.29059829059829057, lift=4.84395061728395)]), RelationRecord(items=frozenset({'mushroom cream sauce', 
'escalope'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mushroom 
cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)]), 
RelationRecord( items=frozenset({'pasta', 'escalope'}), support=0.005865884548726837, ordered_statistics=[
OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, 
lift=4.700811850163794)]), RelationRecord(items=frozenset({'honey', 'fromage blanc'}), support=0.003332888948140248, 
ordered_statistics=[OrderedStatistic(items_ba se=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), 
confidence=0.2450980392156863, lift=5.164270764485569)]), RelationRecord(items=frozenset({'herb & pepper', 
'ground beef '}), support=0.015997866951073192, ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & 
pepper'}), items_add=frozenset({'ground beef'})] 
"""
