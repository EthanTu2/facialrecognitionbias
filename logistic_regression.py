import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# path to the dataset file created by ethan
dataset_path = "dataset.csv"


# these must be updated when ethan finishes the dataset
feature_cols = ["feature_1", "feature_2", "feature_3"] # columns used as input to the model (what the model learns from)
sex_col = "gender" # column the model is trying to predict (target_labels)


def load_data(path):
    # read csv into a table
    df = pd.read_csv(path)

    # select only the feature columns (everything has to be numerical)
    input_data = df[feature_cols].values

    # select the target column (output we want to predict)
    target_labels = df[sex_col].values

    return input_data, target_labels


def train(input_data, target_labels):
    # convert labels like male/female into numbers (0, 1)
    le = LabelEncoder()
    target_encoded = le.fit_transform(target_labels)
    
    # create the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # train the model
    model.fit(input_data, target_encoded)

    return model

    
if __name__ == "__main__":
    # load data
    input_data, target_labels = load_data(dataset_path)
    
    # train model
    model = train(input_data, target_labels)