import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.impute import SimpleImputer, KNNImputer

# Load iris dataset and create DataFrame with missing values
data = sklearn.datasets.load_iris()
X = data.data
y = data.target

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=data.feature_names)

# Add missing values (similar to the example shown)
print("=== Original Data (First 5 Rows) ===")
print(df.head())

# Introduce missing values
print("\n=== Data with Missing Values ===")
df.at[1, df.columns[0]] = np.nan  # sepal length at row 1
df.at[1, df.columns[1]] = np.nan  # sepal width at row 1
df.at[2, df.columns[1]] = np.nan  # sepal width at row 2
df.at[3, df.columns[1]] = 0        # sepal width at row 3 set to 0

print(df.head(5))

# Convert back to numpy array for sklearn imputers
X_missing = df.to_numpy()

print("\n=== Using SimpleImputer ===")
# Option 1: SimpleImputer with mean strategy
imputer1 = SimpleImputer(missing_values=np.nan, strategy="mean")
X_imputed_mean = imputer1.fit_transform(X_missing)

# Set pandas display options for better formatting
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Convert back to DataFrame for display
df_imputed_mean = pd.DataFrame(X_imputed_mean, columns=data.feature_names)
print("SimpleImputer with mean strategy:")
print(df_imputed_mean.head(5))

# Option 2: SimpleImputer with median strategy
print("\nSimpleImputer with median strategy:")
imputer2 = SimpleImputer(missing_values=np.nan, strategy="median")
X_imputed_median = imputer2.fit_transform(X_missing)
df_imputed_median = pd.DataFrame(X_imputed_median, columns=data.feature_names)
print(df_imputed_median.head(5))

# Option 3: SimpleImputer with constant value
print("\nSimpleImputer with constant value (2.0):")
imputer3 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=2.0)
X_imputed_constant = imputer3.fit_transform(X_missing)
df_imputed_constant = pd.DataFrame(X_imputed_constant, columns=data.feature_names)
print(df_imputed_constant.head(5))

print("\n=== Using KNNImputer ===")
# KNNImputer with 2 neighbors
imputer_knn2 = KNNImputer(n_neighbors=2)
X_imputed_knn2 = imputer_knn2.fit_transform(X_missing)
df_imputed_knn2 = pd.DataFrame(X_imputed_knn2, columns=data.feature_names)
print("KNNImputer with n_neighbors=2:")
print(df_imputed_knn2.head(5))

# KNNImputer with 5 neighbors
imputer_knn5 = KNNImputer(n_neighbors=5)
X_imputed_knn5 = imputer_knn5.fit_transform(X_missing)
df_imputed_knn5 = pd.DataFrame(X_imputed_knn5, columns=data.feature_names)
print("\nKNNImputer with n_neighbors=5:")
print(df_imputed_knn5.head(5))

print("\n=== Visual Comparison of Imputation Results ===")
# Compare original and imputed values for the affected rows
comparison = pd.DataFrame({
    'Original': df.iloc[1:3, 0:2].values.flatten(),
    'SimpleImputer(mean)': df_imputed_mean.iloc[1:3, 0:2].values.flatten(),
    'SimpleImputer(median)': df_imputed_median.iloc[1:3, 0:2].values.flatten(),
    'KNNImputer(n=2)': df_imputed_knn2.iloc[1:3, 0:2].values.flatten(),
    'KNNImputer(n=5)': df_imputed_knn5.iloc[1:3, 0:2].values.flatten()
}, index=['Row 1 - Sepal Length', 'Row 1 - Sepal Width', 'Row 2 - Sepal Length', 'Row 2 - Sepal Width'])

print(comparison)

# Reset pandas display options to default
pd.reset_option("display.float_format")