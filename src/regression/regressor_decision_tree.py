#!/usr/bin/env python3

# System integration
import os
from random import random
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

# DT
from sklearn.tree import DecisionTreeRegressor

class DecTreeRegressor(GeneralRegressor):
    
    def __init__(self, max_depth=6):
        """
        Parameters
        -------
        max_depth: Maximum depth of the regression tree. Int values in [1, inf]
        """
        self.max_depth = max_depth

    def create_model(self, X_train, y_train):
        model = DecisionTreeRegressor(max_depth=self.max_depth, 
                                        random_state=24)
        return model
    
    def train_model(self, X_train, y_train, model): 
        model.fit(X_train, y_train)
        return model

    def infer(self, X, model, y_scaler):
        predictions = model.predict(X)
        predictions = np.array(predictions).reshape((len(predictions),1))
        y_pred = rescale(predictions, y_scaler)
        df = pd.DataFrame(y_pred, columns = ['y_pred'])
        return df

    def optimize_hyperparameters(self, X_train, y_train, y_scaler, max_depths=[3,4,8, 12]): 

        # Initialization
        max_depth_opt = self.max_depth
        r2_max = -1e5

        # Run the grid search for hyperparameter optimization
        for max_depth in max_depths:
            self.max_depth = max_depth
            perf = self.cross_validate(X_train, y_train, y_scaler, n_folds=5)[0]
            if perf[0] > r2_max:
                r2_max = perf[0]
                max_depth_opt = max_depth

        print('The best hyperparameters are:') 
        print('max_depth =', max_depth_opt, '         options =', str(max_depths))
        self.max_depth = max_depth_opt
        return
        
    def save_model(self, model, path):
        filename = path + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return 

    def load_model(self, path):
        return load_pickle(path + '.pkl')