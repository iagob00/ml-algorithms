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
df.reset_index(drop=True, inplace=True)
df['Items'] = df['Items'].apply(lambda x: ' '.join(map(str, x)))

# Convert items into binary variables
items_dummies = df['Items'].str.get_dummies(sep=' ')

# Concatenate the binary matrix with the original DataFrame
apriori_ready_df = pd.concat([df['Transaction_ID'], items_dummies], axis=1)
apriori_ready_df.drop('Transaction_ID', axis=1, inplace=True)
# Print the transformed DataFrame
print(apriori_ready_df)


freq = apriori(apriori_ready_df, min_support=0.5, min_confidence=0.20, min_length=2)
rules = list(freq)

for rule in rules:
    pair = rule[0] 
    items = [x for x in pair]
    if len(items) > 1:
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(rule[1]))
        print("Confidence: " + str(rule[2][0][2]))
        print("Lift: " + str(rule[2][0][3]))
        print("=====================================")