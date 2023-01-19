# ai-core-airbnb

So far the data has been cleaned and prepared for the application of linear regression models.

## ```prepare_image_data.py```
This script takes in images from the image folder, records the smallest height of any image in the data set, and then resizes all images such that they all share this height, maintaining the aspect ratio in each case. It discards any images which are not in RGB format.

## ```tabular_data.py```
This script contains functions for cleaning the data and also for preparing the cleand data for the regression model.