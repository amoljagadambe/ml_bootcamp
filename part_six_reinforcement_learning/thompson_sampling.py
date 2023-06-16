# Thompson sampling

# Importing Libraries
import matplotlib.pyplot as plt
import random
import pandas as pd

# Import dataset
dataset = pd.read_csv('data_files/Ads_CTR_Optimisation.csv')

# Implementing thompson sampling
d = dataset.shape[1]  # number of Ads
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_rewards = 0

for n in range(0, dataset.shape[0]):
    max_random = 0
    ad = 0

    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_rewards = total_rewards + reward

print(total_rewards)
print(ads_selected[-10:])
'''
At our last rounds. this algorithm should select optimal ad for this check ads_selected vector
and check the few last entries
print(ads_selected[-10:])
output: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
so from above output we can say ad 4 + 1 is the best one

Output: total rewards:2604  greater than UCB
'''

# Visualizing the result
plt.hist(ads_selected)
plt.title('Histogram ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad selected')
plt.show()  # check the graph folder ads_selections_using_thompson_sampling.png
