import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools


def load_airbnb(clean_df: pd.DataFrame, label):

    df = clean_df.copy()
    labels = np.array(df.pop(label))
    features = np.array(df.select_dtypes(['number']))
    return (features, labels)

def standardise(dataset, mean=None, std=None):
    if mean is None and std is None:
        mean, std = np.mean(dataset, axis=0), np.std(
            dataset, axis=0
        )  # get mean and standard deviation of dataset
    standardized_dataset = (dataset - mean) / std
    return standardized_dataset, (mean, std)


def standardise_multiple(*datasets):
    mean, std = None, None
    for dataset in datasets:
        dataset, (mean, std) = standardise(dataset, mean, std)
        yield dataset

def plot_predictions(y_pred, y_true):
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()

def custom_tune_regression_model_hyperparameters(model_class: type, data: dict, param_dict: dict):

    keys = param_dict.keys()
    combinations = itertools.product(*param_dict.values())
    best_metrics = {"Validation_RMSE":float("inf")}

    def dict_wrapper(combinations):
        for combination in combinations:
            yield dict(zip(keys, combination))

    for attempt in dict_wrapper(combinations):

        model = model_class(**attempt)
        model.fit(data["X_train"], data["y_train"])
        y_hat =  model.predict(data["X_validation"])
        validation_RMSE = metrics.mean_squared_error(data["y_validation"], y_hat, squared=False)

        if validation_RMSE < best_metrics["Validation_RMSE"]:
            best_metrics["Validation_RMSE"] = validation_RMSE
            best_metrics["Validation R^2"] = metrics.r2_score(data["y_validation"], y_hat)
            best_hyperparameters = attempt

    return best_hyperparameters, best_metrics





if __name__ == "__main__":


    np.random.seed(2)
    clean_df = pd.read_csv('./data/clean_tabular_data.csv')
    features, labels = load_airbnb(clean_df, "Price_Night")
    data={}
    #prepearing the data...
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = train_test_split(features, labels, test_size=0.3)
    data["X_validation"], data["X_test"], data["y_validation"], data["y_test"] = train_test_split(data["X_test"], data["y_test"], test_size=0.5)
    data["X_train"], data["X_validation"], data["X_test"] = standardise_multiple(data["X_train"], data["X_validation"], data["X_test"])

    data = {"X_train": data["X_train"], "X_test": data["X_test"], "X_validation": data["X_validation"], "y_train": data["y_train"], "y_test": data["y_test"], "y_validation": data["y_validation"]}


    #fitting the stochastic gradient descent model
    param_dict = {"alpha": [0.00005, 0.0001, 0.0002, 0.0004], "max_iter": [250, 500, 1000]}
    best_hyperparameters, best_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, data, param_dict)
    print(best_hyperparameters)
    print(best_metrics)    