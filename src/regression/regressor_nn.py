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

# Tools
sys.path.append('./../../utils')
from scaling import rescale

# Parent class
from regressor import GeneralRegressor

# Machine Learning tools
from tensorflow.keras import Model, regularizers, losses, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class NeuralNetRegressor(GeneralRegressor):

    def __init__(self, regularizer=0.0, learning_rate=0.005, batchsize=64, epochs=50, patience=10, verbose=0,
                architecture=[32,64,128], hidden_activation_function='relu', output_activation_function='linear'):
        """
        Parameters
        -------
        lam: Regularization coefficient
        lr: Learning rate for weights update
        batchsize: Number of samples drawn at each backpropagation iteration
        epochs: Number of full backpropagation steps
        patience: Number of time steps to wait until an early stoping
        verbosity: Self-explanatory
        architecture: Number of units per layer
        hidden_activation_function: Activation function to use for hidden layers
        output_activation_function: Activation function to use for output layer
        """
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.architecture = architecture
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

    def create_model(self, X_train, y_train):

        # Input layer
        input_layer = tf.keras.Input((X_train.shape[1],))

        # Hidden layers
        hidden_layer = Dense(self.architecture[0], activation=self.hidden_activation_function, kernel_regularizer=regularizers.l2(self.regularizer))(input_layer)
        for i, _ in enumerate(self.architecture[1:]):
            hidden_layer = Dense(self.architecture[i], activation=self.hidden_activation_function, kernel_regularizer=regularizers.l2(self.regularizer))(hidden_layer)

        # Output layer
        output_layer = Dense(y_train.shape[1], activation=self.output_activation_function, kernel_regularizer=regularizers.l2(self.regularizer))(hidden_layer)

        # Instantiate and compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                loss=losses.MeanSquaredError(),
                metrics=['mae'])
        return model
    
    def train_model(self, X_train, y_train, model): 
        model.fit(x=X_train,
                    y=y_train,
                    batch_size=self.batchsize, 
                    epochs=self.epochs, 
                    callbacks=[EarlyStopping(monitor='loss', patience=self.patience)], 
                    verbose = self.verbose)
        return model

    def infer(self, X, model, y_scaler):
        predictions = model.predict(X, verbose=self.verbose)
        y_pred = rescale(predictions, y_scaler)
        df = pd.DataFrame(y_pred, columns = ['y_pred'])
        return df
        
    def optimize_hyperparameters(self, X_train, y_train, y_scaler, regularizers=[0, 0.01, 0.1, 1], learning_rates=[0.001, 0.01, 0.1]): 

        # Initialization
        regularizer_opt = 0.0
        learning_rate_opt = 0.0
        r2_max = -1e5

        # Initialize grid search dataframe
        df_grid_search = pd.DataFrame(columns=['regularizer', 'learning_rate', 'r2'])

        # Run the grid search for hyperparameter optimization
        for regularizer in regularizers:
            for learning_rate in learning_rates:
                self.regularizer = regularizer
                self.learning_rate = learning_rate
                perf = self.cross_validate(X_train, y_train, y_scaler, n_folds=5)[0]
                df_grid_search.loc[len(df_grid_search.index)] = np.array([regularizer, learning_rate, perf[0]])
                if perf[0] > r2_max:
                    r2_max = perf[0]
                    regularizer_opt = regularizer
                    learning_rate_opt = learning_rate

        print('The best hyperparameters are:') 
        print('regularizer =', regularizer_opt, '         options =', str(regularizers))
        print('learning_rate =', learning_rate_opt, '         options =', str(learning_rates))

        self.regularizer = regularizer_opt
        self.learning_rate = learning_rate_opt
        return df_grid_search
        
    def save_model(self, model, path):
        model.save(path)
        return 
    
    def load_model(self, path):
        return models.load_model(path)