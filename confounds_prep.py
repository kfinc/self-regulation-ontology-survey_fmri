# -*- coding: utf-8 -*-


"""
Created on Wed Jul 19 2018
Last edit: Sat Sep 01 2018
@author: kfinc

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


def temp_deriv(dataframe, quadratic = False):
    """Simple function that calculates temporal derivatives for each column of pandas dataframe.

    Parameters
    ----------
    dataframe: pandas dataframe with variable to calculate temporal derivarives

    Returns
    -------
    temp_deriv:  pandas dataframe including original columns and their temporal derivatives ('_td') and (optional)
    their quadratic terms

    """

    temp_deriv = dataframe.copy()

    for col in dataframe.columns:
        #--- backward difference algorithm
        temp = np.diff(dataframe[col], 1, axis = 0)
        temp = np.insert(temp, 0, 0)
        temp = pd.DataFrame(temp)
        temp_deriv[col + '_td'] = temp

    if quadratic == True:
        for col in temp_deriv.columns:
            quad = temp_deriv[col] ** 2
            temp_deriv[col + '_quad'] = quad

    return temp_deriv



def outliers_fd_dvars(dataframe, fd=0.5, dvars=3):
    """Function that calculates motion outliers (frames with frame-wise displacement (FD)
    and DVARS above predefined threshold).

    Parameters
    ----------
    dataframe: pandas dataframe including columns with DVARS and FD
    fd:        threshold for FD (default: 0.5)
    dvars:     threshold for DVARS (+/-SD, default: 3)

    Returns
    -------
    outliers:  pandas dataframe including all outliers datapoints

    """

    df = dataframe.copy()
    df.fillna(value=0, inplace=True)

    dvars_out = np.absolute(df[df.columns[0]].astype(float)) > dvars
    fd_out = df[df.columns[1]].astype(float) > fd

    outliers = (dvars_out == True) | (fd_out == True)
    outliers = pd.DataFrame(outliers.astype('int'))
    outliers.columns = ['scrubbing']

    return outliers
