import numpy as np

# --- Sample Dataset ---
# Columns: [feature1, feature2], last column is label
data = np.array([
    ['sunny', 'hot', 'no'],
    ['sunny', 'hot', 'no'],
    ['overcast', 'hot', 'yes'],
    ['rainy', 'mild', 'yes'],
    ['rainy', 'cool', 'yes'],
    ['rainy', 'cool', 'no'],
    ['overcast', 'cool', 'yes'],
    ['sunny', 'mild', 'no'],
    ['sunny', 'cool', 'yes'],
    ['rainy', 'mild', 'yes']
])

# --- Step 1: Train Naive Bayes ---
def train_naive_bayes(data):
    X = data[:, :-1]
    y = data[:, -1]
    classes = np.unique(y)
    class_probs = {}
    feature_probs = {}
    
    for c in classes:
        X_c = X[y == c]
        class_probs[c] = len(X_c) / len(y)
        feature_probs[c] = []
        for col in range(X.shape[1]):
            values, counts = np.unique(X_c[:, col], return_counts=True)
            probs = {val: count/len(X_c) for val, count in zip(values, counts)}
            feature_probs[c].append(probs)
    return class_probs, feature_probs, classes

# --- Step 2: Predict ---
def predict_naive_bayes(X_test, class_probs, feature_probs, classes):
    predictions = []
    for x in X_test:
        class_scores = {}
        for c in classes:
            score = np.log(class_probs[c])  # use log to prevent underflow
            for i, val in enumerate(x):
                score += np.log(feature_probs[c][i].get(val, 1e-6))  # handle unseen values
            class_scores[c] = score
        predictions.append(max(class_scores, key=class_scores.get))
    return predictions

# --- Step 3: Test the classifier ---
class_probs, feature_probs, classes = train_naive_bayes(data)

# Test data
X_test = np.array([
    ['sunny', 'cool'],
    ['rainy', 'hot'],
    ['overcast', 'mild']
])

predictions = predict_naive_bayes(X_test, class_probs, feature_probs, classes)

# --- Step 4: Print Results ---
for x, p in zip(X_test, predictions):
    print("Input:", x, "-> Predicted class:", p)
