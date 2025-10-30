from itertools import combinations

# --- Step 1: Take transactions from user ---
n = int(input("Enter number of transactions: "))
transactions = []
for i in range(n):
    t = input(f"Enter items in transaction {i+1} (space-separated): ").split()
    transactions.append(t)

# --- Step 2: Get unique items ---
items = sorted(set(i for t in transactions for i in t))

# --- Step 3: Support function ---
def get_support(itemset):
    count = sum(all(i in t for i in itemset) for t in transactions)
    return count / len(transactions)

# --- Step 4: Generate frequent itemsets ---
min_support = 0.3
frequent = {}
for r in range(1, len(items)+1):
    for combo in combinations(items, r):
        sup = get_support(combo)
        if sup >= min_support:
            frequent[combo] = sup

# --- Step 5: Generate association rules ---
min_conf = 0.6
rules = []
for itemset, sup in frequent.items():
    if len(itemset) >= 2:
        for i in range(1, len(itemset)):
            for A in combinations(itemset, i):
                B = tuple(sorted(set(itemset) - set(A)))
                supA = get_support(A)
                supB = get_support(B)
                conf = sup / supA
                lift = conf / supB
                if conf >= min_conf:
                    rules.append((A, B, sup, conf, lift))

# --- Step 6: Sort and display ---
print("\nAssociation Rules (sorted by Lift):")
rules = sorted(rules, key=lambda x: x[4], reverse=True)
for A, B, s, c, l in rules:
    print(f"{A} -> {B} | Support={s:.2f}, Confidence={c:.2f}, Lift={l:.2f}")
