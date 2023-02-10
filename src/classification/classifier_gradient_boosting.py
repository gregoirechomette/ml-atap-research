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
from sklearn.ensemble import GradientBoostingClassifier
from statsmodels.iolib.smpickle import load_pickle
import pickle



class GradBoostingClassifier(GeneralClassifier):
    
    def __init__(self, learning_rate=0.1, n_estimators=100, subsample=0.8, max_depth=4):
        """
        Parameters
        -------
        learning_rate: Shrinks the contribution of each tree by learning_rate. Trade-off to find w/ n_estimators. Float values in [0.0, inf]
        n_estimators: Number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting - large numbers usually result in better performance. Int values in [1, inf]
        subsample: Fraction of samples to be used for fitting the individual base learners. Smaller values lead to a reduction of variance and increase in bias. Float values in [0.0, 1.0]
        max_depth: Maximum depth of the individual regression estimators. Int values in [1, inf]
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth

    def create_model(self, X_train, y_train):
        model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                            n_estimators=self.n_estimators,
                                            subsample=self.subsample,
                                            max_depth=self.max_depth, 
                                            random_state=24)
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

    def optimize_hyperparameters(self, X_train, y_train, learning_rates=[0.05, 0.1, 0.5], subsamples=[0.2, 0.8], max_depths=[4,8]): 

        # Initialization
        learning_rate_opt = self.learning_rate
        subsample_opt = self.subsample
        max_depth_opt = self.max_depth
        acc = -1e5

        # Run the grid search for hyperparameter optimization
        for learning_rate in learning_rates:
            for subsample in subsamples:
                for max_depth in max_depths:
                    self.learning_rate = learning_rate
                    self.subsample = subsample
                    self.max_depth = max_depth
                    perf = self.cross_validate(X_train, y_train, n_folds=5)[0]
                    if perf[0] > acc:
                        acc = perf[0]
                        learning_rate_opt = learning_rate
                        subsample_opt = subsample
                        max_depth_opt = max_depth

        print('The best hyperparameters are:') 
        print('learning_rate =', learning_rate_opt, '         options =', str(learning_rates))
        print('subsample =', subsample_opt, '         options =', str(subsamples))
        print('max_depth =', max_depth_opt, '         options =', str(max_depths))
        
        self.learning_rate = learning_rate_opt
        self.subsample = subsample_opt
        self.max_depth = max_depth_opt
        return
        
    def save_model(self, model, path):
        filename = path + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return 
    
    def load_model(self, path):
        return load_pickle(path + '.pkl')