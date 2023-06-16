# Upper confidence bound
"""
This learning is basically has two things
    1) Exploration
    2) Exploitation

"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Ads_CTR_Optimisation.csv')

# Implementing UCB
d = dataset.shape[1]  # number of Ads
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
'''
for first ten rounds we select 1 ad per round so we can set a upper bound
'''
for n in range(0, dataset.shape[0]):
    max_upper_bound = 0
    ad = 0

    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_rewards = sums_of_rewards[i] / number_of_selections[i]
            '''
            let's calculate the upper bound
            '''
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_rewards + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_rewards = total_rewards + reward

'''
At our last rounds. this algorithm should select optimal ad for this check ads_selected vector
and check the few last entries
print(ads_selected[-10:])
output: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
so from above output we can say ad 4 + 1 is the best one

Output: total rewards:2178
'''

# Visualizing the result
plt.hist(ads_selected)
plt.title('Histogram ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad selected')
plt.show()
