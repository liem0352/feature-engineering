import sklearn.datasets
import openml

print("=== Part 1: sklearn.datasets.load_linnerud ===")

# Exercise 1: Load the linnerud dataset using sklearn.datasets.load_linnerud
linnerud_data = sklearn.datasets.load_linnerud()

# Exercise 2: Check the dataset's data, target, feature_names, target_names
print("Dataset Information:")
print("- Data shape:", linnerud_data.data.shape)
print("- Target shape:", linnerud_data.target.shape)
print("- Feature names:", linnerud_data.feature_names)
print("- Target names:", linnerud_data.target_names)

# Print the full dataset description
print("\nDataset Description:")
print(linnerud_data.DESCR)

# Exercise 3: Read the data and target data into X and y
X = linnerud_data.data
y = linnerud_data.target

# Display the first few rows of X and y for verification
print("\nFirst 5 rows of X:")
print(X[:5])
print("\nFirst 5 rows of y:")
print(y[:5])

print("\n=== Part 2: OpenML Diabetes Dataset ===")

# Load diabetes dataset from OpenML
datasets = openml.datasets.list_datasets(output_format='dataframe')

# Find diabetes dataset ID
diabetes_dataset = datasets[datasets['name'] == 'diabetes']
if not diabetes_dataset.empty:
    diabetes_id = int(diabetes_dataset.iloc[0]['did'])
    print(f"Diabetes dataset ID: {diabetes_id}")
else:
    print("Diabetes dataset not found")
    exit()

# Load the diabetes dataset
diabetes = openml.datasets.get_dataset(diabetes_id)

# Print dataset details
print("\nDiabetes description:")
print(diabetes.description)

print("\nDiabetes URL:")
print(diabetes.url)

# Get data and target
data, target, categorical_indicator, attribute_names = diabetes.get_data(
    dataset_format='dataframe',
    target='class'
)

print("\nDiabetes data:")
print(data)

print("\nDiabetes target:")
print(target)