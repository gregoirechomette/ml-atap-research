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

# LR
import sklearn.linear_model

class LinearRegressor(GeneralRegressor):
    
    def __init__(self):
        return

    def create_model(self, X_train, y_train):
        model = sklearn.linear_model.LinearRegression()
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
        
    def save_model(self, model, path):
        filename = path + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return 
    
    def load_model(self, path):
        return load_pickle(path + '.pkl')