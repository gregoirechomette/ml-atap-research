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
from sklearn.tree import DecisionTreeClassifier
from statsmodels.iolib.smpickle import load_pickle
import pickle


class DecTreeClassifier(GeneralClassifier):
    
    def __init__(self, max_depth=6):
        """
        Parameters
        -------
        max_depth: Maximum depth of the regression tree. Int values in [1, inf]
        """
        self.max_depth = max_depth

    def create_model(self, X_train, y_train):
        model = DecisionTreeClassifier(max_depth=self.max_depth, 
                                        random_state=24)
        return model
    
    def train_model(self, model, X_train, y_train): 
        model.fit(X_train, y_train)
        return model

    def infer(self, X, model):
        predictions = model.predict_proba(X)[:,1:2]
        predictions = np.array(predictions).reshape((len(predictions),1))
        df = pd.DataFrame(predictions, columns = ['y_pred'])
        return df

    def optimize_hyperparameters(self, X_train, y_train, max_depths=[3,4,8, 12]): 

        # Initialization
        max_depth_opt = self.max_depth
        acc_max = -1e5

        # Run the grid search for hyperparameter optimization
        for max_depth in max_depths:
            self.max_depth = max_depth
            perf = self.cross_validate(X_train, y_train, n_folds=5)[0]
            if perf[0] > acc_max:
                acc_max = perf[0]
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