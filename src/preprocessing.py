import numpy as np
from sklearn.model_selection import train_test_split

def flatten_images(images, labels):
    """
    This function that an array of images; flattens them into 1D vectors;
    split the data into training, validation and test sets; normalize these sets and finally returns them
    """
    X = images.reshape(images.shape[0], -1) # Flattens the images
    y = labels.reshape(labels.shape[0], -1) # Reshapes the labels
    return X, y

def split_data(X, y, random_state=42):
    """This function that an array of images; flattens them into 1D vectors;
    split the data into training, validation and test sets"""

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    return  X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean() # Calculates the mean of the training set
    std = X_train.std() # Calculates the standard deviation of the the training set

    # Normalize all sets
    X_train = (X_train - mean)/std 
    X_val = (X_val - mean)/std
    X_test = (X_test - mean)/std

    return X_train, X_val, X_test, mean, std


