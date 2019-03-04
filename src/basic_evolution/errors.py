from math import sqrt

import numpy as np


def error_dtw_all(forecast, observations):
    '''
    Calculate DTW metric of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    # distance, path = fastdtw(forecast.hsig_series, observations, dist=euclidean)
    distance = 1
    # print("DTW")
    return distance


def error_rmse_all(forecast, observations):
    '''
    Calculate RMSE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    penalty_var = abs((np.var(observations) - np.var(forecast.hsig_series)) / np.var(observations)) + 1

    result = 0.0

    for pred, obs in zip(forecast.hsig_series, observations):
        result += pow(pred - obs, 2)

    return sqrt(result / len(observations)) * penalty_var


def error_rmse_peak(forecast, observations):
    '''
    Calculate peakwise RMSE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    peak_thr = np.mean(observations)

    result = 0.0
    points = 0
    for pred, obs in zip(forecast.hsig_series, observations):

        if obs >= peak_thr:
            result += pow(pred - obs, 2)
            points += 1

    return sqrt(result / points)


def error_mae_peak(forecast, observations):
    '''
    Calculate peakwise MAE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    peak_thr = np.mean(observations)

    result = 0.0
    points = 0
    for pred, obs in zip(forecast.hsig_series, observations):

        if obs >= peak_thr:
            result += abs(pred - obs)
            points += 1

    return (result / points)


def error_mae_all(forecast, observations):
    pred = np.array(forecast.hsig_series)
    obs = observations[:len(forecast.hsig_series)]

    return np.sum(np.abs(pred - obs)) / len(observations)
