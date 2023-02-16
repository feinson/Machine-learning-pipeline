import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import yaml
from data_handling import *
from itertools import count, product
import time

from sklearn import model_selection
from sklearn import metrics

class AirbnbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(torch.nn.Module):

    default_hyperparams = {"optimiser": "Adam", "learning_rate": 0.001, "width":32, "depth":5}

    def __init__(self, hyperparams_dict=default_hyperparams):
        super(NeuralNetwork, self).__init__()

        width, depth = hyperparams_dict["width"], hyperparams_dict["depth"]
        n_hidden = (depth - 2)

        self.layers = torch.nn.Sequential(torch.nn.Linear(11, width),           #creates a neural network based on the width and depth
        *(torch.nn.ReLU(), torch.nn.Linear(width, width))*(n_hidden//2),        #tuple unpacking is used to create the desried number of layers
        *(torch.nn.ReLU(),)*(n_hidden%2),
        torch.nn.Linear(width, 1))

        self.lr = hyperparams_dict["learning_rate"]                             #sets learning rate
        exec(f'self.optimiser = torch.optim.{hyperparams_dict["optimiser"]}')   #sets optimiser
        print("used dict")


    def forward(self, x):
        return self.layers(x)

def train(model, train_data, validation_data=None, num_epochs=30, batch_size=8, print_out=False, timed=True, write_loss=False):
    if timed == True:
        start_time = time.time()

    if write_loss == True:
        writer = SummaryWriter()
        train_idx, validation_idx = count(1), count(1)

    model.train()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if validation_data is not None:
        validation_loader = DataLoader(validation_data, batch_size=batch_size)
    
    optimiser = model.optimiser(model.parameters(), lr=model.lr)

    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if write_loss == True:
                writer.add_scalar("Training_Loss", loss.item(), next(train_idx))
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        if validation_data is not None:
            for inputs, targets  in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if write_loss == True:
                    writer.add_scalar("Validation_Loss", loss.item(), next(validation_idx))
        
        if print_out == True:
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss/len(train_loader):.4f}")

    return model, (time.time() - start_time) if timed == True else None


def custom_tune_nn_hyperparameters(data: dict, param_dict_list: list):

    best_metrics = {"Validation_RMSE":float("inf")}
    train_data = AirbnbDataset(data["X_train"], data["y_train"])
    for param_dict in param_dict_list:
        keys = param_dict.keys()
        combinations = product(*param_dict.values())        #itertools product
        

        for attempt in (dict(zip(keys, combination)) for combination in combinations):

            model = NeuralNetwork(attempt)
            #upto
            trained_model = train(model, train_data, num_epochs=30, print_out=True)

            with torch.no_grad():
                trained_model.eval() # set model to evaluation mode
                inputs = torch.tensor(data["X_validation"], dtype=torch.float32)
                y_hat = trained_model(inputs).numpy()
                validation_RMSE = metrics.mean_squared_error(data["y_validation"], y_hat, squared=False)

            if validation_RMSE < best_metrics["Validation_RMSE"]:
                best_metrics["Validation_RMSE"] = validation_RMSE
                best_metrics["Validation_R^2"] = metrics.r2_score(data["y_validation"], y_hat)
                best_hyperparameters = attempt

    return best_hyperparameters, best_metrics



if __name__ == "__main__":

    
    np.random.seed(2)
    clean_df = pd.read_csv('.//data//clean_tabular_data.csv')
    features, labels = load_airbnb(clean_df, label="Price_Night")
    data={}
    #preparing the data... I like to put everything in a dictionary.
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = model_selection.train_test_split(features, labels, test_size=0.3)
    data["X_validation"], data["X_test"], data["y_validation"], data["y_test"] = model_selection.train_test_split(data["X_test"], data["y_test"], test_size=0.5)
    data["X_train"], (train_mean, train_std) = standardise(data["X_train"])
    data["X_validation"], data["X_test"] = standardise_multiple(data["X_validation"], data["X_test"], mean=train_mean, std=train_std)

# Create data loaders for the training, validation, and test sets
    train_data = AirbnbDataset(data["X_train"], data["y_train"])
    validation_data = AirbnbDataset(data["X_validation"], data["y_validation"])

    with open("nn_configuration_file.yaml", "r") as stream:
        try:
            p = yaml.safe_load(stream)
            print(p)
        except yaml.YAMLError as exc:
            print(exc)

    trained_model, _ = train(NeuralNetwork(p[0]), train_data, validation_data=validation_data, num_epochs=30, print_out=True)
    print(isinstance(trained_model, torch.nn.Module))
    


    with torch.no_grad():
        trained_model.eval() # set model to evaluation mode
        inputs = torch.tensor(data["X_validation"], dtype=torch.float32)
        predictions = trained_model(inputs).numpy()
        validation_RMSE = metrics.mean_squared_error(data["y_validation"], predictions, squared=False)
        print(validation_RMSE)