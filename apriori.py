from itertools import combinations

# --- Sample Dataset ---
transactions = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['bread', 'milk', 'nuts'],
    ['milk', 'bread', 'apple'],
    ['bread', 'apple'],
    ['milk', 'bread', 'apple', 'nuts']
]

# --- Step 1: Find all frequent itemsets ---
def get_frequent_itemsets(transactions, min_support):
    num_transactions = len(transactions)
    
    # Count individual items
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] = item_counts.get(frozenset([item]), 0) + 1

    # Filter by minimum support
    frequent_itemsets = {item: count/num_transactions for item, count in item_counts.items() if count/num_transactions >= min_support}
    all_frequent = dict(frequent_itemsets)
    
    k = 2
    while frequent_itemsets:
        # Generate candidate sets of size k
        candidates = [keys[i].union(keys[j]) 
                      for i, keys_i in enumerate(list(frequent_itemsets.keys()))
                      for j, keys_j in enumerate(list(frequent_itemsets.keys())[i+1:])
                      if len(keys_i.union(keys_j)) == k]
        
        # Count support for candidates
        candidate_counts = {c: 0 for c in candidates}
        for t in transactions:
            t_set = set(t)
            for c in candidates:
                if c.issubset(t_set):
                    candidate_counts[c] += 1
        
        # Keep only those above min_support
        frequent_itemsets = {c: count/num_transactions for c, count in candidate_counts.items() if count/num_transactions >= min_support}
        all_frequent.update(frequent_itemsets)
        k += 1

    return all_frequent

# --- Step 2: Generate Association Rules ---
def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    confidence = support / frequent_itemsets.get(antecedent, 1)
                    if confidence >= min_confidence:
                        rules.append((set(antecedent), set(consequent), confidence))
    return rules

# --- Run Apriori ---
min_support = 0.4
min_confidence = 0.6

frequent_itemsets = get_frequent_itemsets(transactions, min_support)
rules = generate_rules(frequent_itemsets, min_confidence)

# --- Display Results ---
print("=== Frequent Itemsets ===")
for itemset, support in frequent_itemsets.items():
    print(list(itemset), "=> support:", round(support, 2))

print("\n=== Association Rules ===")
for antecedent, consequent, confidence in rules:
    print(f"{antecedent} -> {consequent} (confidence: {round(confidence, 2)})")
