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
from classification_tools import compute_classification_metrics


class GeneralClassifier:
    
    def __init__(self):
        return
    

    @abstractmethod
    def create_model(self, X_train, y_train):
        """
        Description: Method to create a classification model
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
            Classification model defined and compiled
        """
        pass


    @abstractmethod
    def train_model(self, X_train, y_train, model):
        """
        Description: Method to train a classification model
        ----------
        
        Parameters
        -------
        X_train : Pandas dataframe (N_points x N_features)
            Dataset input for training
        y_train : Pandas dataframe (N_points x 1)
            Dataset output for training
        model : Scipy, sklearn, or tensorflow model
            Classification model defined and compiled 

        Returns
        -------
        model : Scipy, sklearn, or tensorflow model
            Classification model defined, compiled, and trained
        """
        pass


    @abstractmethod
    def infer(self, X, model):
        """
        Description: Method to predict an output given a trained model
        ----------
        
        Parameters
        -------
        X : Pandas dataframe (N_points x N_features)
            Dataset input for inference
        model : Scipy, sklearn, or tensorflow model
            Classification model defined, compiled, and trained

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
            Classification model defined, compiled, and trained
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
            Classification model defined, compiled, and trained
        """
        pass


    def train_and_infer(self, X_train, y_train, X_eval):
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
            Classification model defined, compiled, and trained
        y_test_pred: Pandas dataframe (N_points x 1)
            Predictions in a dataframe format
        """
        model = self.create_model(X_train, y_train)
        model = self.train_model(model, X_train, y_train)
        y_eval_pred = self.infer(X_eval, model)
        return model, y_eval_pred
    

    def cross_validate(self, X, y, n_folds=3):
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
            All possible metrics to cross validate the classification performance
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
            X_val = np.array(split_X[i])
            y_val = np.array(split_y[i])

            # Obtain the training set for this cross validation fold
            X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=0)
            y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=0)

            # Re-convert numpy arrays to pandas DF
            X_train = pd.DataFrame(data = X_train, columns = X_column_names)
            X_val = pd.DataFrame(data = X_val, columns = X_column_names)
            y_train = pd.DataFrame(data = y_train, columns = y_column_names)
            y_val = pd.DataFrame(data = y_val, columns = y_column_names)

            # Predict and output results of regression
            _, y_val_pred = self.train_and_infer(X_train, y_train, X_val)
            cross_val_metrics[i:i+1,:] = compute_classification_metrics(y_val, y_val_pred).reshape((1,6))

        # Select the row with median accuracy
        median_row = cross_val_metrics[cross_val_metrics[:,0] == np.median(cross_val_metrics, axis=0)[0]][0,:].reshape((6,))

        return median_row, np.mean(cross_val_metrics, axis=0)


def test_all_models(X, y, hyperparameter_opt=True):
    """
    Description: Method to cross-validate the training of the models
    ----------
    
    Parameters
    -------
    X : 2D numpy array (N_points x n_features)
        Dataset input to evaluate all the models
    y : 2D numpy array (N_points x n_labels)
        Dataset output to evaluate all the models
        

    Returns
    -------
    df_models_comparison : pandas dataframe
        A dataframe with all the models and the associated cross validated classification metrics
    """

    # Import the classifier classes
    from classifier_logistic import LogisticClassifier
    from classifier_decision_tree import DecTreeClassifier
    from classifier_random_forest import RandForestClassifier
    from classifier_gradient_boosting import GradBoostingClassifier
    from classifier_nn import NeuralNetClassifier

    # Create the pandas dataframe to populate with evaluation metrics
    df_models_comparison = pd.DataFrame(columns=['Accuracy', 
                                                'Precision', 
                                                'True Positive Rate', 
                                                'False Positive Rate', 
                                                'True Negative Rate', 
                                                'False Negative Rate'])

    # Logistic
    print('Logistic')
    classifier_logistic = LogisticClassifier()
    df_models_comparison.loc[len(df_models_comparison.index)] = classifier_logistic.cross_validate(X,y)[0]

    # Decision tree
    print()
    print('Decision Tree')
    classifier_dt = DecTreeClassifier()
    if hyperparameter_opt:
        classifier_dt.optimize_hyperparameters(X,y)
    df_models_comparison.loc[len(df_models_comparison.index)] = classifier_dt.cross_validate(X,y)[0]

    # Random forest
    print()
    print('Random forest')
    classifier_rf = RandForestClassifier()
    if hyperparameter_opt:
        classifier_rf.optimize_hyperparameters(X,y)
    df_models_comparison.loc[len(df_models_comparison.index)] = classifier_rf.cross_validate(X,y)[0]

    # Gradient boosting
    print()
    print('Gradient boosting')
    classifier_gb = GradBoostingClassifier()
    if hyperparameter_opt:
        classifier_gb.optimize_hyperparameters(X,y)
    df_models_comparison.loc[len(df_models_comparison.index)] = classifier_gb.cross_validate(X,y)[0]

    # Neural network
    print()
    print('Neural network')
    classifier_nn = NeuralNetClassifier()
    if hyperparameter_opt:
        classifier_nn.optimize_hyperparameters(X,y)
    df_models_comparison.loc[len(df_models_comparison.index)] = classifier_nn.cross_validate(X,y)[0]

    # Add a column with the names of all the models at the very left
    df_models_comparison.insert(0, 'Model', ['Logisitic', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Net'])

    return df_models_comparison