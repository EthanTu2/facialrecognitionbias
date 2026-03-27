import pandas as pd
import numpy as np

# Load FairFace CSV
df = pd.read_csv("train_labels.csv")

# Use gender as the target label
df["true_label"] = df["gender"]

# Make random results the same every time the code runs
np.random.seed(42)

# Define race-specific noise levels
race_noise = {
    "White": 0.80,
    "Middle Eastern": 0.90,
    "East Asian": 1.00,
    "Latino_Hispanic": 1.10,
    "Indian": 1.20,
    "Southeast Asian": 1.25,
    "Black": 1.35
}

# Number of features
n_features = 32

# Generate synthetic feature vectors
feature_rows = []

for _, row in df.iterrows():
    gender = row["true_label"]
    race = row["race"]
    noise = race_noise.get(race, 1.0)

    # Set class center based on gender
    if gender == "Male":
        center = np.ones(n_features)
    else:
        center = -1 * np.ones(n_features)

    # Add race-specific noise
    features = center + np.random.normal(0, noise, n_features)
    feature_rows.append(features)

# Convert features to a DataFrame
feature_df = pd.DataFrame(feature_rows, columns=[f"feature_{i+1}" for i in range(n_features)])

# Combine metadata and features
final_df = pd.concat([df[["file", "age", "gender", "race", "true_label"]].reset_index(drop=True),feature_df.reset_index(drop=True)], axis=1)

# Save to CSV
final_df.to_csv("feature_dataset.csv", index=False)

# Quick check
print(final_df.head())
print(final_df.shape)