import pandas as pd
from collections import defaultdict

# --- Step 1: Dataset ---
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

# --- Step 2: Train Naive Bayes ---
def train_naive_bayes(df, target):
    model = defaultdict(lambda: defaultdict(dict))
    classes = df[target].unique()
    for c in classes:
        subset = df[df[target] == c]
        model['priors'][c] = len(subset) / len(df)
        for feature in df.columns.drop(target):
            for v in df[feature].unique():
                model[feature][v][c] = len(subset[subset[feature] == v]) / len(subset)
    return model

# --- Step 3: Predict and display intermediate steps ---
def predict(model, sample):
    classes = model['priors'].keys()
    probs = {}
    print("\n=== Probability Computation Steps ===")
    for c in classes:
        prob = model['priors'][c]
        print(f"\nClass = {c}, Prior = {prob:.3f}")
        for feature, value in sample.items():
            cond_prob = model[feature].get(value, {}).get(c, 0)
            print(f"P({feature}={value} | {c}) = {cond_prob:.3f}")
            prob *= cond_prob
        probs[c] = prob
        print(f"→ Combined likelihood for class '{c}' = {prob:.6f}")
    return max(probs, key=probs.get), probs

# --- Step 4: Build model & Predict ---
model = train_naive_bayes(df, 'PlayTennis')

sample = {}
for f in df.columns[:-1]:
    sample[f] = input(f"Enter {f}: ").strip().capitalize()

pred, probs = predict(model, sample)

print("\n=== Final Results ===")
print("Posterior Probabilities:", probs)
print("Predicted Class:", pred)
