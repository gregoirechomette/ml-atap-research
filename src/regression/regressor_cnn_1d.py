#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# ML utils
from tensorflow.keras import Model, regularizers, losses, models
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class CNN1DRregressor:

    def __init__(self):
        return
    
    def create_model(self, X_train, y_train, 
                    architecture_filters=[32,64], 
                    architecture_dense = [64, 32],
                    hidden_activation_function='relu', 
                    output_activation_function='linear',
                    lam=0.0, 
                    lr=0.005):

        """
        Description: Method to create a fully-connected tensorflow neural network model
        ----------
        
        Parameters
        -------
        X_train : 2D numpy array (N_points x N_features)
            Typical dataset input 
        y_train : 2D numpy array (N_points x N_labels)
            Typical dataset output 
        architecture: 1D list of integers
            Number of filters for each layer
        hidden_activation_function: string
            Non-linear activation function to use for each hidden layer
        output_activation_function: string
            Non-linear activation function to use for the final layer
        lam: double
            Regularization parameter to address over-fitting issues
        lr: double
            Learning rate used in the gradient descent optimization

        Returns
        -------
        model : tensorflow model
            fully-connected neural network model defined and compiled
        """

        # Input layer
        input_layer = tf.keras.Input(X_train[0,:,:].shape)

        # Hidden layers - convolution
        hidden_layer = Conv1D(architecture_filters[0], kernel_size=3, activation=hidden_activation_function, kernel_regularizer=regularizers.l2(lam))(input_layer)
        hidden_layer = MaxPooling1D(pool_size=2)(hidden_layer)

        for i, _ in enumerate(architecture_filters[1:]):
            hidden_layer = Conv1D(architecture_filters[i], kernel_size=3, activation=hidden_activation_function, kernel_regularizer=regularizers.l2(lam))(hidden_layer)
            hidden_layer = MaxPooling1D(pool_size=2)(hidden_layer)

        # Hidden layers - fully connected
        hidden_layer = Flatten()(hidden_layer)
        for i, _ in enumerate(architecture_dense):
            hidden_layer = Dense(architecture_dense[i], activation=hidden_activation_function, kernel_regularizer=regularizers.l2(lam))(hidden_layer)

        output_layer = Dense(y_train.shape[1], activation=output_activation_function, kernel_regularizer=regularizers.l2(lam))(hidden_layer)

        # Instantiate and compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=lr),
                loss=losses.MeanSquaredError(),
                metrics=['mae'])

        return model


    
    def train_model(self, model, X_train, y_train, batchsize=64, epochs=100, patience=10, verbosity=0): 
    
        """
        Description: Method to train the model
        ----------
        
        Parameters
        -------
        model : tensorflow model
            neural network model defined and compiled
        X_train : 2D numpy array (N_points x N_features)
            Training dataset input 
        y_train : 2D numpy array (N_points x N_labels)
            Training dataset output 
        batchsize : int
            Number of points selected at each gradient descent step
        epochs : int
            Number of gradient descent steps involving the whole input points
        patience : int 
            Early stopping argument
        verbosity : int
            Quantity of information printed on the terminal during the training process
    
        Returns
        -------
        model : tensorflow model
            Trained neural network model
        """

        # Training function call
        model.fit(x=X_train,
                    y=y_train,
                    batch_size=batchsize, 
                    epochs=epochs, 
                    callbacks=[EarlyStopping(monitor='loss', patience=patience)], 
                    verbose = verbosity)
        
        return model
        
    def optimize_hyperparameters(self, X_val, y_val): 

        """
        Description: Method to optimize the hyperparameters
        ----------
        
        Parameters
        -------
        X_val : 2D numpy array (N_points x N_features)
            Validation dataset input 
        y_val : 2D numpy array (N_points x N_labels)
            Validation dataset output 
    
        Returns
        -------
        lambda_opt : double
            Optimal regularizer
        lr_opt : double
            Optimal learning rate
        """

        # Generate an error message
        print('Info: optimize_hyperparameters function not implemented')
        raise ValueError('optimize_hyperparameters function not implemented')
        return
        
    
    def save_model(self, model, path):

        """
        Description: Method to save a model in a pre-specified path
        ----------
        
        Parameters
        -------
        model : tensorflow model
            Neural network model to save
        path : string
           Path to the repository where the model needs to be saved
        """

        model.save(path)
        return 
    
    def load_model(self, path):

        """
        Description: Load a model in a pre-specified path
        ----------
        
        Parameters
        -------
        path : string
           Path to the repository where the model needs to be loaded from

        Returns
        -------
        model : tensorflow model
            Neural network model
        """

        return models.load_model(path)


    def infer(self, X, model):

        """
        Description: Predict the outputs given an input and the trained model
        ----------
        
        Parameters
        -------
        X : 2D numpy array (N_points x N_features)
            Input set to predict
        model : tensorflow model
            Neural network model
        
        Returns
        -------
        y : 2D numpy array (N_points x N_labels)
            Predicted output 
        """

        return model.predict(X)