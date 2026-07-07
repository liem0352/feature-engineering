import numpy as np
import pandas as pd
import sklearn.datasets

# Load iris dataset for demonstration
data = sklearn.datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("Original DataFrame:")
print(df.head())
print(f"\nShape: {df.shape}")

# Add missing values (NaN)
print("\n--- Adding Missing Values ---")
df.at[1, df.columns[0]] = np.nan  # Set sepal length at row 1 to NaN
df.at[1, df.columns[1]] = np.nan  # Set sepal width at row 1 to NaN
df.at[2, df.columns[1]] = np.nan  # Set sepal width at row 2 to NaN
df.at[3, df.columns[1]] = 0        # Set sepal width at row 3 to 0 (not missing, just a zero)

print(df.head())

# Check for missing values
print("\n--- Checking for Missing Values ---")
print("Missing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Handle missing values (options)
print("\n--- Handling Missing Values ---")

# Option 1: Drop rows with any missing values
df_dropna = df.dropna()
print(f"After dropping rows with missing values: {df_dropna.shape}")

# Option 2: Fill missing values with mean
print("\nFilling missing values with column means:")
df_fill_mean = df.fillna(df.mean())
print(df_fill_mean.head())

# Option 3: Fill missing values with specific value
df_fill_zero = df.fillna(0)
print("\nFilling missing values with 0:")
print(df_fill_zero.head())