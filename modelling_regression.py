import pandas as pd
import numpy as np

#Import models to test
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression

#Import plotting
import matplotlib.pyplot as plt

#Import other useful
from sklearn import metrics
import itertools
import os

#Import other libraries in this repository
import hyperparams_configuration_file as hcf
from data_handling import *


def plot_predictions(y_pred, y_true):
    """
    It creates a simple plot of the predicted values, vs the true values.
    """

    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()

def custom_tune_regression_model_hyperparameters(model_class: type, data: dict, param_dict_list: list):

    best_metrics = {"Validation_RMSE":float("inf")}
    for param_dict in param_dict_list:
        keys = param_dict.keys()
        combinations = itertools.product(*param_dict.values()) # Generate all possible combiations for the grid search.
        

        for attempt in (dict(zip(keys, combination)) for combination in combinations):

            model = model_class(**attempt)
            model.fit(data["X_train"], data["y_train"])
            y_hat = model.predict(data["X_validation"])
            validation_RMSE = metrics.mean_squared_error(data["y_validation"], y_hat, squared=False)

            if validation_RMSE < best_metrics["Validation_RMSE"]:
                best_metrics["Validation_RMSE"] = validation_RMSE
                best_metrics["Validation_R^2"] = metrics.r2_score(data["y_validation"], y_hat)
                best_hyperparameters = attempt

    return best_hyperparameters, best_metrics

def save_all_models_and_find_best(models_and_params, data, save_best_model=True):
    overall_best_RMSE = float("inf")

    for model_pair in models_and_params.items():

        best_hyperparameters, best_metrics = custom_tune_regression_model_hyperparameters(model_pair[0], data, model_pair[1])
        if best_metrics["Validation_RMSE"] < overall_best_RMSE:
            overall_best_RMSE = best_metrics["Validation_RMSE"]
            overall_best_model_name = model_pair[0].__name__
            overall_best_hyperparameters = best_hyperparameters

        print(f"{model_pair[0].__name__}:", best_hyperparameters, best_metrics)
        the_model = model_pair[0](**best_hyperparameters)
        the_model.fit(data["X_train"], data["y_train"])
        if save_best_model == True:
            save_model(the_model, hyperparams=best_hyperparameters, metrics=best_metrics, folder=f"models{os.sep}regression{os.sep}{model_pair[0].__name__[:-9]}")
    
    print("_____________________________________")
    print(f"The best model is {overall_best_model_name} with hyperparamters given by {overall_best_hyperparameters}. It has a Validation set RMSE of {overall_best_RMSE}.")
    return overall_best_model_name, overall_best_hyperparameters, overall_best_RMSE
    

if __name__ == "__main__":

    np.random.seed(2)
    data = prepare_data("Price_Night")


    
    #evaluating and saving all models
    models_and_params = {SGDRegressor: hcf.SGDRegressor_params,
                        LinearRegression: hcf.LinearRegression_params, 
                        DecisionTreeRegressor: hcf.DecisionTreeRegressor_params,
                        RandomForestRegressor: hcf.RandomForestRegressor_params,
                        GradientBoostingRegressor: hcf.GradientBoostingRegressor_params}
    
    save_all_models_and_find_best(models_and_params, data)

