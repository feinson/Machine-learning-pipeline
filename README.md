# ai-core-airbnb

This was a project I completed for the AiCore data science specialism course. It involves processing tabular and image data, including data cleaning in Pandas, followed by building a pipeline to train and test various machine learning models to make predictions from the data. The files ```modelling_regression.py``` and ```modelling_classification.py``` contain pipelines for testing various models from SKlearn. This works by performing a grid search to find the best hyperparameters for each model, and then comparing the different models to each other. The ```neural_network.py``` functions similarly, but this time a custom neural network is built and trained in PyTorch, according to the hyperparameters.
Note that the image data, and tensorboard runs are gitignored.

## Summary of what I have learnt
    - How to clean and wrangle data with Pandas.
    - How to process image data and tabular data to prepare it for machine learning tasks.
    - How to use SKlearn to train and test various machine learning models. I also have a good understanding of how models like random forests and decision trees work under the hood, although this is not neccesary for the project.
    - How to perform a grid search to tune hyperparmeters.
    - How to create and configure a neural network using PyTorch.
    - How to track the training of machine learning models using tensorboard.

 Individually the scripts function as follows.
## ```tabular_data.py```
This script loads in the unclean data as a csv file, and uses Pandas to fix problems with the data and saves the cleaned data as another csv. A .py file is used, but in a real-world situation a Jupyter notebook would be preferable, so that the data doesn't have to be loaded every time


For the next two scripts, the task is to create models to predict the nightly price of AirBnBs, based on the tabular data.
## ```modelling_regression.py```
This script performs a grid-search to tune the hyperparameters of five machine-learning models as provided by SKLearn. Specifically, ```LinearRegression``` (OLS), ```SGDRegressor```, ```DecisionTreeRegressor```, ```GradientBoostingRegressor``` and ```RandomForestRegressor```. It saves each tuned and trained model in the directory ```./data/models/regression```. It also compares all five models to each other and determines the best one. The hyperparameters to test are stored in ```hyperparams_configuration_file.py```.

## ```neural_network.py```
This script performs a grid-search to tune the hyperparameters of a neural-network created with PyTorch. The hyperparameters include the width and depth of the neural-network. The grid-search takes in the hyperparameters to test from a YAML file ```nn_configuration_file.yaml```. Of course, the hyperparameters could be just as easily be stored in a .py file or a JSON file.

## ```airbnb_bedrooms.py```
This script aims to demonstrate the robustness of the pipeline by repeating the task but with the goal of predicting the number of bedrooms in the AirBnB. It imports from ```modelling_regression.py``` and ```neural_network.py```.

## ```modelling_classification.py```
This script functions similiarly to the regression version, but this time the machine learning task is to predict the 'Category' of the AirBnB. The categories are 'Treehouses', 'Chalets', 'Amazing Pools', 'Offbeat' and 'Beachfront'. The hyperparameters to test are stored in ```hyperparams_configuration_file.py```.


## ```prepare_image_data.py```
This script takes in images of the AirBnB listings from the image folder (hidden with gitignore), records the smallest height of any image in the data set, and then resizes all images such that they all share this height, maintaining the aspect ratio in each case. It discards any images which are not in RGB format. The processed images are saved in the ```all_processed_images``` folder (gitignored). An extension task might be to create a convolutional neural network to make predictions about the listings based on these images.
