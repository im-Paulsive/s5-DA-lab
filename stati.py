import pandas as pd
import numpy as np
from scipy import stats

# --- Step 1: Read data from CSV ---
file = input("Enter CSV file path: ")
df = pd.read_csv(file)

print("\n=== Statistical Description of Data ===")
print(df.describe(include='all'))

# --- Step 2: Choose numeric column ---
print("\nAvailable numeric columns:", list(df.select_dtypes(include=np.number).columns))
col = input("Enter column name to analyze: ")

data = df[col].dropna()

# --- Step 3: Compute Quartiles and IQR ---
Q1 = data.quantile(0.25)
Q2 = data.quantile(0.5)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# --- Step 4: Find Outliers using IQR ---
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data < lower_bound) | (data > upper_bound)]

print(f"\nQ1: {Q1:.2f}, Q2 (Median): {Q2:.2f}, Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
print("\nOutliers:")
print(outliers if not outliers.empty else "No outliers found")

# --- Step 5: Remove outliers ---
clean_data = data[(data >= lower_bound) & (data <= upper_bound)]

# --- Step 6: Compute Statistics on Clean Data ---
mean = clean_data.mean()
median = clean_data.median()
mode = list(clean_data.mode())

std_dev = clean_data.std()

print("\n=== After Removing Outliers ===")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode(s): {mode}")
print(f"Standard Deviation: {std_dev:.2f}")

# --- Step 7: Check if Unimodal, Bimodal, or Trimodal ---
num_modes = len(mode)
if num_modes == 1:
    mode_type = "Unimodal"
elif num_modes == 2:
    mode_type = "Bimodal"
elif num_modes == 3:
    mode_type = "Trimodal"
else:
    mode_type = f"Multimodal ({num_modes} modes)"

print(f"Data Type: {mode_type}")
