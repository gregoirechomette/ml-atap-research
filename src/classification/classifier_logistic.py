#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd

# Parent class
from classifier import GeneralClassifier

# LR utils
import sklearn.linear_model
import pickle
from statsmodels.iolib.smpickle import load_pickle


class LogisticClassifier(GeneralClassifier):
    
    def __init__(self):
        pass

    def create_model(self, X_train, y_train):
        model = sklearn.linear_model.LogisticRegression()
        return model
    
    def train_model(self, model, X_train, y_train): 
        if not (isinstance(y_train, (np.ndarray, np.generic))):
            y_train = y_train.to_numpy()
        model.fit(X_train, y_train.ravel())
        return model

    def infer(self, X, model):
        predictions = model.predict_proba(X)[:,1:2]
        predictions = np.array(predictions).reshape((len(predictions),1))
        df = pd.DataFrame(predictions, columns = ['y_pred'])
        return df
        
    def save_model(self, model, path):
        filename = path + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return 
    
    def load_model(self, path):
        return load_pickle(path + '.pkl')