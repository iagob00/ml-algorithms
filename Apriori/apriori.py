import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori


data = {
    'Transaction_ID': [1, 2, 3, 4],
    'Items': [['apple', 'banana', 'orange'],
              ['apple', 'grapes', 'banana'],
              ['grapes', 'orange'],
              ['apple', 'orange', 'banana', 'grapes']]
}

df = pd.DataFrame(data)

df['Items'] = df['Items'].apply(lambda x: ' '.join(map(str, x)))

# Convert items into binary variables
items_dummies = df['Items'].str.get_dummies(sep=' ')

# Concatenate the binary matrix with the original DataFrame
apriori_ready_df = pd.concat([df['Transaction_ID'], items_dummies], axis=1)

# Print the transformed DataFrame
print(apriori_ready_df)