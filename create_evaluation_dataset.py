import pandas as pd
import numpy as np

# Load FairFace CSV
df = pd.read_csv("train_labels.csv")

# Use gender as the ground truth label
df["true_label"] = df["gender"]

# Define race-specific accuracy to simulate bias
race_accuracy = {
    "White": 0.95,
    "East Asian": 0.80,
    "Southeast Asian": 0.75,
    "Middle Eastern": 0.85,
    "Latino_Hispanic": 0.78,
    "Indian": 0.72,
    "Black": 0.70
}

# Generate predicted labels based on race-specific accuracy
def make_prediction(row):
    true = row["true_label"]
    race = row["race"]
    acc = race_accuracy.get(race, 0.80)

    # With probability = acc, prediction is correct
    if np.random.rand() < acc:
        return true
    else:
        # Otherwise flip the label
        return "Male" if true == "Female" else "Female"

# Generate predicted labels for each row
df["predicted_label"] = df.apply(make_prediction, axis=1)

# Mark whether each prediction matches the true label (1 = correct, 0 = incorrect)
df["correct"] = (df["true_label"] == df["predicted_label"]).astype(int)

# Keep only necessary columns for evaluation
eval_df = df[["file", "age", "gender", "race","true_label", "predicted_label", "correct"]]

# Save to CSV
eval_df.to_csv("evaluation_dataset.csv", index=False)

# Quick sanity check (not saved in dataset)
print("\nOverall accuracy:")
print((eval_df["true_label"] == eval_df["predicted_label"]).mean())

print("\nAccuracy by race:")
print(eval_df.groupby("race").apply(lambda x: (x["true_label"] == x["predicted_label"]).mean()))