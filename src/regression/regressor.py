#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

from abc import abstractmethod

# Standard python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Regression tools
sys.path.append('./../../utils')
from scaling import rescale
from regression_tools import compute_regression_metrics


class GeneralRegressor:
    
    def __init__(self):
        return


    @abstractmethod
    def create_model(self, X_train, y_train):
        """
        Description: Method to create a regression model
        ----------
        
        Parameters
        -------
        X_train : Pandas dataframe (N_points x N_features)
            Typical dataset input for training
        y_train : Pandas dataframe (N_points x 1)
            Typical dataset output for training

        Returns
        -------
        model : Scipy, sklearn, or tensorflow model
            Regression model defined and compiled
        """
        pass


    @abstractmethod
    def train_model(self, X_train, y_train, model):
        """
        Description: Method to train a regression model
        ----------
        
        Parameters
        -------
        X_train : Pandas dataframe (N_points x N_features)
            Dataset input for training
        y_train : Pandas dataframe (N_points x 1)
            Dataset output for training
        model : Scipy, sklearn, or tensorflow model
            Regression model defined and compiled 

        Returns
        -------
        model : Scipy, sklearn, or tensorflow model
            Regression model defined, compiled, and trained
        """
        pass


    @abstractmethod
    def infer(self, X, model, y_scaler):
        """
        Description: Method to predict an output given a trained model
        ----------
        
        Parameters
        -------
        X : Pandas dataframe (N_points x N_features)
            Dataset input for inference
        model : Scipy, sklearn, or tensorflow model
            Regression model defined, compiled, and trained

        Returns
        -------
        y : Pandas dataframe (N_points x 1)
            Predictions in a dataframe format
        """
        pass


    @abstractmethod
    def save_model(self, model, path):
        """
        Description: Method to save a model in a pre-specified path
        ----------
        
        Parameters
        -------
        model : Scipy, sklearn, or tensorflow model
            Regression model defined, compiled, and trained
        path : Scipy, sklearn, or tensorflow model
            Full path to where to save the model, without the extension

        Returns
        -------
        None
        """
        pass


    @abstractmethod
    def load_model(self, path):
        """
        Description: Method to load a model from a pre-specified path
        ----------
        
        Parameters
        -------
        path : Scipy, sklearn, or tensorflow model
            Full path to where to save the model, without the extension

        Returns
        -------
        model : Scipy, sklearn, or tensorflow model
            Regression model defined, compiled, and trained
        """
        pass


    def train_and_infer(self, X_train, y_train, X_eval, y_scaler):
        """
        Description: Method to train a model with a training set and infer on a test set
        ----------
        
        Parameters
        -------
        X_train : Pandas dataframe (N_points x N_features)
            Dataset input for training
        y_train : Pandas dataframe (N_points x 1)
            Dataset output for training
        X_train : Pandas dataframe (N_points x N_features)
            Dataset input for testing

        Returns
        -------
        model : Scipy, sklearn, or tensorflow model
            Regression model defined, compiled, and trained
        y_test_pred: Pandas dataframe (N_points x 1)
            Predictions in a dataframe format
        """
        model = self.create_model(X_train, y_train)
        model = self.train_model(X_train, y_train, model)
        y_eval_rescaled_pred = self.infer(X_eval, model, y_scaler)
        return model, y_eval_rescaled_pred


    def cross_validate(self, X, y, y_scaler, n_folds=5):
        """
        Description: Method to cross-validate the training of the models
        ----------
        
        Parameters
        -------
        X : Pandas dataframe (N_points x n_features)
            Dataset input for the cross validation
        y : Pandas dataframe (N_points x 1)
            Dataset output for the cross validation
            
    
        Returns
        -------
        [R2, MRE, MedRE, MSE, MAE, MedAE] : list
            All possible metrics to cross validate the regression performance
        """

        # Save the dataframe column names
        X_column_names = list(X.columns)
        y_column_names = list(y.columns)

        # Shuffle the data
        X, y = shuffle(X, y, random_state=0)
        
        # Split data into different folds
        split_X = np.array_split(X, n_folds, axis=0)
        split_y = np.array_split(y, n_folds, axis=0)

        # Prepare the metrics 
        cross_val_metrics = np.zeros((n_folds,6))

        # Entering cross validation loop
        for i in range(n_folds):

            # Obtain the validation set for this cross validation fold
            X_eval = np.array(split_X[i])
            y_eval = np.array(split_y[i])

            # Obtain the training set for this cross validation fold
            X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=0)
            y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=0)

            # Re-convert numpy arrays to pandas DF
            X_train = pd.DataFrame(data = X_train, columns = X_column_names)
            X_eval = pd.DataFrame(data = X_eval, columns = X_column_names)
            y_train = pd.DataFrame(data = y_train, columns = y_column_names)
            y_eval = pd.DataFrame(data = y_eval, columns = y_column_names)

            # Trains and outputs results of regression, already rescales
            _, y_eval_rescaled_pred = self.train_and_infer(X_train, y_train, X_eval, y_scaler)

            # Rescale the label
            y_eval_rescaled = rescale(y_eval, y_scaler)

            # Compute the regression metrics
            cross_val_metrics[i:i+1,:] = compute_regression_metrics(y_eval_rescaled, y_eval_rescaled_pred).reshape((1,6))

        return np.median(cross_val_metrics, axis=0), np.mean(cross_val_metrics, axis=0)


def test_all_models(X, y, y_scaler, hyperparameter_opt=True):
    """
    Description: Method to cross-validate the training of the models
    ----------
    
    Parameters
    -------
    X : Pandas dataframe (N_points x n_features)
        Dataset input to evaluate all the models
    y : Pandas dataframe (N_points x 1)
        Dataset output to evaluate all the models
        
    Returns
    -------
    df_models_comparison : pandas dataframe
        A dataframe with all the models and the associated cross validated regression metrics
    """

    # Import the regressor classes
    from regressor_linear import LinearRegressor
    from regressor_linear_robust import LinearRobustRegressor
    from regressor_decision_tree import DecTreeRegressor
    from regressor_random_forest import RandForestRegressor
    from regressor_gradient_boosting import GradBoostingRegressor
    from regressor_nn import NeuralNetRegressor

    # Create the pandas dataframe to populate with evaluation metrics
    df_models_comparison = pd.DataFrame(columns=['R2', 'Mean_RE', 'Med_RE', 'Mean_SE', 'Mean_AE', 'Med_AE'])

    # Linear
    print('Linear')
    regressor_linear = LinearRegressor()
    df_models_comparison.loc[len(df_models_comparison.index)] = regressor_linear.cross_validate(X,y,y_scaler)[0]

    # Decision tree
    print()
    print('Decision tree')
    regressor_dt = DecTreeRegressor()
    if hyperparameter_opt:
        regressor_dt.optimize_hyperparameters(X,y,y_scaler)
    df_models_comparison.loc[len(df_models_comparison.index)] = regressor_dt.cross_validate(X,y,y_scaler)[0]

    # Random forest
    print()
    print('Random forest')
    regressor_rf = RandForestRegressor()
    if hyperparameter_opt:
        regressor_rf.optimize_hyperparameters(X,y,y_scaler)
    df_models_comparison.loc[len(df_models_comparison.index)] = regressor_rf.cross_validate(X,y,y_scaler)[0]

    # Gradient boosting
    print()
    print('Gradient boosting')
    regressor_gb = GradBoostingRegressor()
    if hyperparameter_opt:
        regressor_gb.optimize_hyperparameters(X,y,y_scaler)
    df_models_comparison.loc[len(df_models_comparison.index)] = regressor_gb.cross_validate(X,y,y_scaler)[0]

    # Neural network
    print()
    print('Neural network')
    regressor_nn = NeuralNetRegressor()
    if hyperparameter_opt:
        regressor_nn.optimize_hyperparameters(X,y,y_scaler)
    df_models_comparison.loc[len(df_models_comparison.index)] = regressor_nn.cross_validate(X,y,y_scaler)[0]

    # Add a column with the names of all the models at the very left
    df_models_comparison.insert(0, 'Model', ['Linear', 'Decision tree', 'Random forest', 'Gradient boosting', 'Neural network'])

    return df_models_comparison