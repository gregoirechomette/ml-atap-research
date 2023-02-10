#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd


def compute_regression_metrics(y_true, y_pred):

    """
    Description: Method to compute all the regression evaluation metrics
    ----------
    
    Parameters
    -------
    y_true : 2D numpy array (N_points x 1)
        Ground truth vector 
    y_pred : 2D numpy array (N_points x 1)
        Regressor predictions

    Returns
    -------
    [R2, MRE, MedRE, MSE, MAE, MedAE] : list
        All possible metrics to evaluate the regression performance
    """

    # Correct the ground truth to avoid divisions by zero
    y_true_corr = np.where(np.absolute(y_true) < 1e-5, 1e-5, y_true)


    # Compute the R2
    r_squared = np.where(np.sum(np.square(y_true - np.mean(y_true, axis=0))) > 0,
                    1 - (np.sum(np.square(y_pred - y_true))/np.sum(np.square(y_true - np.mean(y_true, axis=0)))),
                    0)
    
    # Compute the mean relative errror
    mean_re = np.mean(np.absolute((y_pred - y_true)/y_true_corr), axis=0)

    # Compute the median relative error
    median_re = np.median(np.absolute((y_pred - y_true)/y_true_corr), axis=0)
    
    # Compute the mean squared error
    mean_se = np.mean(np.square(y_pred - y_true), axis=0)

    # Compute the mean absolute error
    mean_ae = np.mean(np.absolute(y_pred - y_true), axis=0)

    # Compute the median absolute error
    median_ae = np.median(np.absolute(y_pred - y_true), axis=0)

    # Agglomerate all the metrics in one
    regression_metrics = np.array([np.round(r_squared, decimals=2)[0], 
                                    np.round(mean_re, decimals=2)[0], 
                                    np.round(median_re, decimals=2)[0],
                                    np.format_float_scientific(mean_se, precision=2), 
                                    np.format_float_scientific(mean_ae, precision=2), 
                                    np.format_float_scientific(median_ae, precision=2)])

    return regression_metrics

def  compute_regression_errors(y_true, y_pred):

    """
    Description: Method to compute all the regression evaluation metrics
    ----------
    
    Parameters
    -------
    y_true : 2D numpy array (N_points x 1)
        Ground truth vector 
    y_pred : 2D numpy array (N_points x 1)
        Regressor predictions

    Returns
    -------
    a_errors : 2D numpy array (N_points x 1)
        Absolute errors
    r_errors : 2D numpy array (N_points x 1)
        Relative errors in % (corrected by 1e-5 if 0 in denominator)
    """

    # Correct the ground truth to avoid divisions by zero
    y_true_corr = np.where(np.absolute(y_true) < 1e-5, 1e-5, y_true)

    # Compute the absolute errors
    a_errors = np.absolute(y_pred - y_true)

    # Compute the relative errors
    r_errors = 100 * np.absolute((y_pred - y_true)/y_true_corr)

    return a_errors, r_errors