import modelling_regression
import neural_network
import data_handling
import hyperparams_configuration_file as hcf
import yaml

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression


if __name__ == "__main__":

    data = data_handling.prepare_data("bedrooms")
    
    models_and_params = {SGDRegressor: hcf.SGDRegressor_params,
                        LinearRegression: hcf.LinearRegression_params, 
                        DecisionTreeRegressor: hcf.DecisionTreeRegressor_params,
                        RandomForestRegressor: hcf.RandomForestRegressor_params,
                        GradientBoostingRegressor: hcf.GradientBoostingRegressor_params}

    sk_learn_name, sk_hyperparameters, sk_RMSE =  modelling_regression.save_all_models_and_find_best(models_and_params, data, save_best_model=False)

    with open("nn_configuration_file.yaml", "r") as f:
        try:
            nn_params_list_dicts = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


    nn_hyperparameters, nn_metrics = neural_network.custom_tune_nn_hyperparameters(data, nn_params_list_dicts)
    nn_RMSE = nn_metrics["Validation_RMSE"]

    print("Overall scoring:")
    print("________________________________________________________________________________")
    if nn_RMSE < sk_RMSE:
        print(f"The best model was the neural network with hypeparameters given by {nn_hyperparameters}.\n It has a validation set RMSE of {nn_RMSE}.")
    else:
        print(f"The best model was the SKLearn {sk_learn_name} with hyperparameters given by {sk_hyperparameters}.\n It has a validation set RMSE of {sk_RMSE}")