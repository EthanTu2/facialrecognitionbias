import pandas as pd

# Load the feature dataset
df = pd.read_csv("feature_dataset.csv")

# Make random results the same every time the code runs
random_state = 42

# Define race groups
races = [
    "White",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino_Hispanic",
    "Indian",
    "Black"
]

# Check how many samples are available for each race
print("Available samples by race:")
print(df["race"].value_counts())
print()

# Dataset A: balanced (same number of samples for each race)
n_A = 100

dataset_A = pd.concat([
    df[df["race"] == race].sample(n=n_A, random_state=random_state)
    for race in races
])

dataset_A = dataset_A.sample(frac=1, random_state=random_state).reset_index(drop=True)
dataset_A.to_csv("dataset_A_balanced.csv", index=False)

# Dataset B: moderately skewed (white has more samples than other groups)
counts_B = {
    "White": 200,
    "East Asian": 80,
    "Southeast Asian": 80,
    "Middle Eastern": 80,
    "Latino_Hispanic": 80,
    "Indian": 80,
    "Black": 80
}

dataset_B = pd.concat([
    df[df["race"] == race].sample(n=counts_B[race], random_state=random_state)
    for race in races
])

dataset_B = dataset_B.sample(frac=1, random_state=random_state).reset_index(drop=True)
dataset_B.to_csv("dataset_B_skewed.csv", index=False)

# Dataset C: strongly imbalanced (white dominates the dataset)
counts_C = {
    "White": 300,
    "East Asian": 40,
    "Southeast Asian": 40,
    "Middle Eastern": 40,
    "Latino_Hispanic": 40,
    "Indian": 40,
    "Black": 40
}

dataset_C = pd.concat([
    df[df["race"] == race].sample(n=counts_C[race], random_state=random_state)
    for race in races
])

dataset_C = dataset_C.sample(frac=1, random_state=random_state).reset_index(drop=True)
dataset_C.to_csv("dataset_C_imbalanced.csv", index=False)

# Print summary
print("Dataset A shape:", dataset_A.shape)
print(dataset_A["race"].value_counts())
print()

print("Dataset B shape:", dataset_B.shape)
print(dataset_B["race"].value_counts())
print()

print("Dataset C shape:", dataset_C.shape)
print(dataset_C["race"].value_counts())