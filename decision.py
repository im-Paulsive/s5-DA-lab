import numpy as np

# --- Sample Dataset ---
# Columns: [feature1, feature2], last column is label
data = np.array([
    [1, 2, 0],
    [1, 4, 0],
    [2, 3, 0],
    [3, 2, 1],
    [3, 4, 1],
    [4, 3, 1]
])

# --- Step 1: Entropy Function ---
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# --- Step 2: Information Gain ---
def info_gain(parent, left, right):
    total_len = len(left) + len(right)
    gain = entropy(parent) - (len(left)/total_len)*entropy(left) - (len(right)/total_len)*entropy(right)
    return gain

# --- Step 3: Best Split ---
def best_split(X, y):
    best_feature, best_value, best_gain = None, None, -1
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for v in values:
            left_idx = X[:, feature] <= v
            right_idx = X[:, feature] > v
            if sum(left_idx) == 0 or sum(right_idx) == 0:
                continue
            gain = info_gain(y, y[left_idx], y[right_idx])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = v
    return best_feature, best_value, best_gain

# --- Step 4: Build Tree Recursively ---
def build_tree(X, y, depth=0, max_depth=3):
    # If all labels are same or max depth reached, return leaf
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return {'label': np.bincount(y).argmax()}
    
    feature, value, gain = best_split(X, y)
    if gain == 0:
        return {'label': np.bincount(y).argmax()}
    
    left_idx = X[:, feature] <= value
    right_idx = X[:, feature] > value
    
    left_tree = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right_tree = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    
    return {'feature': feature, 'value': value, 'left': left_tree, 'right': right_tree}

# --- Step 5: Prediction ---
def predict(tree, x):
    if 'label' in tree:
        return tree['label']
    if x[tree['feature']] <= tree['value']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)

# --- Build Decision Tree ---
X = data[:, :-1]
y = data[:, -1].astype(int)
tree = build_tree(X, y)

# --- Test Predictions ---
print("Decision Tree:", tree)
print("Predictions:")
for i in range(len(X)):
    print(X[i], "->", predict(tree, X[i]))
