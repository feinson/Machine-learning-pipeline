import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
from data_handling import *

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
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train(model, train_data, num_epochs=30, batch_size=8, lr=0.001):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        

        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss/len(train_loader):.4f}")

    return model



if __name__ == "__main__":

    
    np.random.seed(2)
    clean_df = pd.read_csv('.//data//clean_tabular_data.csv')
    features, labels = load_airbnb(clean_df, label="Price_Night")
    data={}
    #preparing the data...
    data["X_train"], data["X_test"], data["y_train"], data["y_test"] = model_selection.train_test_split(features, labels, test_size=0.3)
    data["X_validation"], data["X_test"], data["y_validation"], data["y_test"] = model_selection.train_test_split(data["X_test"], data["y_test"], test_size=0.5)
    data["X_train"], (train_mean, train_std) = standardise(data["X_train"])
    data["X_validation"], data["X_test"] = standardise_multiple(data["X_validation"], data["X_test"], mean=train_mean, std=train_std)

# Create data loaders for the training, validation, and test sets
    train_data = AirbnbDataset(data["X_train"], data["y_train"])
    validation_data = AirbnbDataset(data["X_validation"], data["y_validation"])

    trained_model = train(NeuralNetwork(), train_data, num_epochs=30)


    with torch.no_grad():
        trained_model.eval() # set model to evaluation mode
        inputs = torch.tensor(data["X_validation"], dtype=torch.float32)
        predictions = trained_model(inputs).numpy()
        validation_RMSE = metrics.mean_squared_error(data["y_validation"], predictions, squared=False)
        print(validation_RMSE)