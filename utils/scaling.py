#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

def find_scaler(X):
    X_scaler = MinMaxScaler()
    X_scaler.fit(X)
    return X_scaler

def normalize(X, X_scaler):
    X_normalized = X_scaler.transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    return X_normalized

def rescale(X, X_scaler):
    X_rescaled = X_scaler.inverse_transform(X)
    return X_rescaled

def save_scaler(scaler, path):
    dump(scaler, open(path + '.pkl', 'wb'))
    return

def load_scaler(path):
    scaler = load(open(path + '.pkl', 'rb'))
    return scaler