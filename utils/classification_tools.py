#!/usr/bin/env python3

# System integration
import os
import sys
import abc
import json

# Standard python libraries
import numpy as np
import pandas as pd


def compute_classification_metrics(y_true, y_pred, threshold=0.5):
        """
        Description: Method to compute all the regression evaluation metrics
        ----------
        
        Parameters
        -------
        y_true : 2D numpy array (N_points x 1)
            Ground truth vector 
        y_pred : 2D numpy array (N_points x 1)
            Regressor predictions
        threshold: limit between the two classes in the predictions
    
        Returns
        -------
        [accuracy, fpr, fnr] : list
            All possible metrics to evaluate the classification performance
        """

        # Introduce a new variable to assign the classes
        y_pred_classified = np.array(y_pred)

        # Transform probabilities into classes predictions
        y_pred_classified[y_pred_classified < threshold] = 0
        y_pred_classified[y_pred_classified >= threshold] = 1

        # Total positives and negatives
        tot_neg = np.count_nonzero(y_true == np.zeros(y_true.shape)) 
        tot_pos = np.count_nonzero(y_true == np.ones(y_true.shape))
        pos_predicted = np.count_nonzero(y_pred_classified == np.ones(y_pred_classified.shape))

        assert tot_neg > 0, 'Only one class, can not compute metrics'
        assert tot_pos > 0, 'Only one class, can not compute metrics'

        # Compute the metrics
        accuracy = (np.count_nonzero(y_pred_classified == y_true))/y_pred_classified.shape[0]
        if pos_predicted == 0:
            precision = 0.0
        else:
            precision = (np.count_nonzero(np.logical_and(y_pred_classified == y_true, y_true == np.ones(y_true.shape))))/pos_predicted
        true_pos_rate = np.count_nonzero(np.logical_and(y_pred_classified == y_true, y_true == np.ones(y_true.shape)))/tot_pos
        false_pos_rate = np.count_nonzero(np.logical_and(y_pred_classified != y_true, y_true == np.zeros(y_true.shape)))/tot_neg
        true_neg_rate = np.count_nonzero(np.logical_and(y_pred_classified == y_true, y_true == np.zeros(y_true.shape)))/tot_neg
        false_neg_rate = np.count_nonzero(np.logical_and(y_pred_classified != y_true, y_true == np.ones(y_true.shape)))/tot_pos

        # Agglomerate all the metrics in one
        classification_metrics = np.array([np.round(accuracy * 100, decimals=1),
                                        np.round(precision * 100, decimals=1),
                                        np.round(true_pos_rate * 100, decimals=1),
                                        np.round(false_pos_rate * 100, decimals=1) ,
                                        np.round(true_neg_rate * 100, decimals=1), 
                                        np.round(false_neg_rate * 100, decimals=1)])

        return classification_metrics