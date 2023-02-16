import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn import model_selection


def save_model(model, hyperparams, metrics, folder, name="model"):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    if isinstance(model, torch.nn.Module):
        model_path = os.path.join(folder, f"{name}.pt")
        torch.save(model.state_dict(), model_path)
    elif isinstance(model, sklearn.base.BaseEstimator):
        model_path = os.path.join(folder, f"{name}.joblib")
        joblib.dump(model, model_path)  #save model
    else:
        raise TypeError("Input model must be either an SKlearn model, or a PyTorch model.")

    hyperparams_path = os.path.join(folder, f"{name}_hyperparams.json")
    metrics_path = os.path.join(folder, f"{name}_metrics.json")

    with open(hyperparams_path, 'w') as fp: #save hyperparmeters of model
        json.dump(hyperparams, fp)

    with open(metrics_path, 'w') as fp:      #save metrice of model
        json.dump(metrics, fp)

def load_airbnb(clean_df: pd.DataFrame, label):

    df = clean_df.copy()
    labels = np.array(df.pop(label))
    try:
        df[label]
        return None
    except:
        features = np.array(df.select_dtypes(['number']))
    return (features, labels)


def standardise(dataset, mean=None, std=None):
    if mean is None and std is None:
        mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)  # get mean and standard deviation of dataset
    standardised_dataset = (dataset - mean) / std
    return standardised_dataset, (mean, std)

def standardise_multiple(*datasets, mean=None, std=None):
    for dataset in datasets:
        dataset, (_, _) = standardise(dataset, mean, std)
        yield dataset


def prepare_data(label:str):
    clean_df = pd.read_csv('.//data//clean_tabular_data.csv')
    features, labels = load_airbnb(clean_df, label=label)
    data={}
    #preparing the data...
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = model_selection.train_test_split(features, labels, test_size=0.3)
    data["X_validation"], data["X_test"], data["y_validation"], data["y_test"] = model_selection.train_test_split(data["X_test"], data["y_test"], test_size=0.5)
    data["X_train"], (train_mean, train_std) = standardise(data["X_train"])
    data["X_validation"], data["X_test"] = standardise_multiple(data["X_validation"], data["X_test"], mean=train_mean, std=train_std)
    return data


if __name__ == "__main__":
    pass