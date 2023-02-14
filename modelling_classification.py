import pandas as pd
import numpy as np
import math

#Import models to test
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


#Import other useful
from sklearn import metrics
from sklearn import model_selection
import itertools
import os

#Import other libraries in this repository
import hyperparams_configuration_file as hcf
from data_handling import *

def custom_tune_classification_model_hyperparameters(model_class: type, data: dict, param_dict_list: list):

    best_metrics = {"Validation_accuracy":0.0}
    for param_dict in param_dict_list:
        keys = param_dict.keys()
        combinations = itertools.product(*param_dict.values())
        

        for attempt in (dict(zip(keys, combination)) for combination in combinations):
            model = model_class(**attempt)
            model.fit(data["X_train"], data["y_train"])
            y_hat =  model.predict(data["X_validation"])
            validation_accuracy = metrics.accuracy_score(data["y_validation"], y_hat)

            if validation_accuracy > best_metrics["Validation_accuracy"]:
                best_metrics["Validation_accuracy"] = validation_accuracy
                best_metrics["Validation_f1"] = metrics.f1_score(data["y_validation"], y_hat, average="weighted")
                best_metrics["Validation_precison"] = metrics.precision_score(data["y_validation"], y_hat, average="weighted", zero_division=0)
                best_metrics["Validation_recall"] =metrics.recall_score(data["y_validation"], y_hat, average="weighted")
                best_hyperparameters = attempt
    return best_hyperparameters, best_metrics


def save_all_models_and_find_best(models_and_params, data):
    overall_best_accuracy = 0

    for model_pair in models_and_params.items():

        best_hyperparameters, best_metrics = custom_tune_classification_model_hyperparameters(model_pair[0], data, model_pair[1])
        if best_metrics["Validation_accuracy"] > overall_best_accuracy:
            overall_best_accuracy = best_metrics["Validation_accuracy"]
            overall_best_model_name = model_pair[0].__name__
            overall_best_hyperparameters = best_hyperparameters

        print(f"{model_pair[0].__name__}:", best_hyperparameters, best_metrics)
        the_model = model_pair[0](**best_hyperparameters)
        the_model.fit(data["X_train"], data["y_train"])
        save_model(the_model, hyperparams=best_hyperparameters, metrics=best_metrics, folder=f"models{os.sep}classification{os.sep}{model_pair[0].__name__[:-10]}")

    print("_____________________________________")
    print(f"The best model is {overall_best_model_name} with hyperparamters given by {overall_best_hyperparameters}. It has a Validation set accuracy of {overall_best_accuracy}.")


if __name__ == "__main__":

    np.random.seed(2)
    clean_df = pd.read_csv('.//data//clean_tabular_data.csv')
    features, labels = load_airbnb(clean_df, label="Category")
    data={}
    #preparing the data...
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = model_selection.train_test_split(features, labels, test_size=0.3)
    data["X_validation"], data["X_test"], data["y_validation"], data["y_test"] = model_selection.train_test_split(data["X_test"], data["y_test"], test_size=0.5)
    data["X_train"], (train_mean, train_std) = standardise(data["X_train"])
    data["X_validation"], data["X_test"] = standardise_multiple(data["X_validation"], data["X_test"], mean=train_mean, std=train_std)


    models_and_params = {LogisticRegression: hcf.LogisticRegression_params, 
                        DecisionTreeClassifier: hcf.DecisionTreeClassifier_params,
                        RandomForestClassifier: hcf.RandomForestClassifier_params,
                        GradientBoostingClassifier: hcf.GradientBoostingClassifier_params}
    
    save_all_models_and_find_best(models_and_params, data)
    
   