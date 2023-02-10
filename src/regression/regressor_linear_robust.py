#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd

# Tools
sys.path.append('./../../utils')
import pickle
from scaling import rescale
from statsmodels.iolib.smpickle import load_pickle

# Parent class
from regressor import GeneralRegressor

# LRR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor


class LinearRobustRegressor(GeneralRegressor):
    
    def __init__(self, min_samples=0.95, max_trials=100):
        """
        Parameters
        -------
        min_samples: Minimum number of samples chosen randomly from original data in fraction form. Float value in [0,1] 
        max_trials: Maximum number of iterations for random sample selection
        """
        self.min_samples = min_samples
        self.max_trials = max_trials

    def create_model(self, X_train, y_train):
        model = RANSACRegressor(LinearRegression(),
                                min_samples=self.min_samples,
                                max_trials=self.max_trials,
                                loss='absolute_error')
        return model
    
    def train_model(self, X_train, y_train, model): 
        model.fit(X_train, y_train)
        return model

    def infer(self, X, model, y_scaler):
        predictions = model.predict(X)
        y_pred = rescale(predictions, y_scaler)
        df = pd.DataFrame(y_pred, columns = ['y_pred'])
        return df

    def optimize_hyperparameters(self, X_train, y_train, y_scaler, min_samples_list=[3,4,8, 12]): 

        # Initialization
        min_samples_opt = self.min_samples
        r2_max = -1e5

        # Run the grid search for hyperparameter optimization
        for min_samples in min_samples_list:
            self.min_samples = min_samples
            perf = self.cross_validate(X_train, y_train, y_scaler, n_folds=5)[0]
            if perf[0] > r2_max:
                r2_max = perf[0]
                min_samples_opt = min_samples

        print('The best hyperparameters are:') 
        print('min_samples =', min_samples_opt, '         options =', str(min_samples_list))
        self.min_samples = min_samples_opt
        return
        
    def save_model(self, model, path):
        filename = path + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return 
    
    def load_model(self, path):
        return load_pickle(path + '.pkl')