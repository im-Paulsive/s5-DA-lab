import pandas as pd
import numpy as np
from math import log2
#from sklearn.tree import DecisionTreeClassifier, plot_tree
#import matplotlib.pyplot as plt

df = pd.read_csv("exp11.csv")
target_column = df.columns[-1]
features = list(df.columns[:-1])

def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -sum(p * log2(p) for p in probabilities if p > 0)

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for val, count in zip(values, counts):
        subset = data[data[feature] == val]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (count / len(data)) * subset_entropy
    gain = total_entropy - weighted_entropy
    return gain

def build_tree(data, features, target):
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    if len(features) == 0:
        return data[target].mode()[0]
    gains = {feature: information_gain(data, feature, target) for feature in features}
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}
    for val in np.unique(data[best_feature]):
        subset = data[data[best_feature] == val]
        subtree = build_tree(
            subset.drop(columns=[best_feature]),
            [f for f in features if f != best_feature],
            target
        )
        tree[best_feature][val] = subtree
    return tree

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→", tree)
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}[{feature} = {val}]")
            print_tree(subtree, indent + "  ")

def predict(tree, sample):
    # If we reached a leaf node (string or number)
    if not isinstance(tree, dict):
        return tree

    # Get current feature to check
    feature = next(iter(tree))
    feature_value = sample.get(feature)

    # Move down the correct branch
    if feature_value in tree[feature]:
        # Recursively predict within this branch
        return predict(tree[feature][feature_value], sample)
    else:
        # If we don’t find a matching branch, 
        # but the current node is already a leaf (like 'Yes' or 'No'),
        # return that leaf value instead of 'Unknown'.
        sub_tree = tree[feature]
        # If *all* branches under this feature lead to the same label,
        # we can safely return that label.
        if all(not isinstance(v, dict) and v == list(sub_tree.values())[0] for v in sub_tree.values()):
            return list(sub_tree.values())[0]
        else:
            return "Unknown"



overall_entropy = entropy(df[target_column])
print(f"Total Entropy of dataset (INFO(D)): {overall_entropy:.4f}\n")

gains = {}
for feature in features:
    gain = information_gain(df, feature, target_column)
    gains[feature] = gain
    print(f"Information Gain for '{feature}': {gain:.4f}")

best_feature = max(gains, key=gains.get)
print(f"\nFeature with highest Information Gain: '{best_feature}'\n")

decision_tree = build_tree(df, features, target_column)
print("Decision Tree (Text Format):")
print_tree(decision_tree)


# --- Take user input as a dictionary ---
sample = {}
for f in features:
    val = input(f"Enter " + f + ": ").strip().capitalize()
    sample[f] = val

# --- Predict the class ---
prediction = predict(decision_tree, sample)
print("\nPredicted class:", prediction)



#X = pd.get_dummies(df[features])
#y = df[target_column]

#clf = DecisionTreeClassifier(criterion="entropy")
#clf.fit(X, y)

#plt.figure(figsize=(12,8))
#plot_tree(clf, feature_names=X.columns, class_names=clf.classes_,
 #         filled=True, rounded=True, fontsize=10)
#plt.title("Decision Tree")
#plt.show()
