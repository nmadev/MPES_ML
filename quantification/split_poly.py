from numpy.core.numeric import Infinity
from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np
import seaborn as sns
from data_io_split import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_regression(temp_train, temp_test, target_train, target_test, n):
    models = []
    predicted = []
    scores = []
    # initializes, trains, and tests the trained model finding a percent error
    for i in range(3):
        models.append(Pipeline([('poly', PolynomialFeatures(degree=n[i])), ('linear', LinearRegression(fit_intercept=False))]))
        models[i].fit(temp_train[i], target_train[i])
        predicted.append(models[i].predict(temp_test[i]))
        scores.append(percent_error(target_test[i], predicted[i]))
    
    # returns the percent error of the model
    return ave_multiple_arr(scores)

if __name__ == '__main__':
    # finds algorithm name and makes a folder to hold images if the folder does not already exist
    path_name = sys.argv[0][:-3]
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG+NH3"]
    num_trials = 100
    highest_degree = 3

    # import data into pandas data frame
    dfs = load_data()

    # randomly samples 80% of the data for training and 20% for testing
    traindf = []
    testdf = []
    for it in range(num_trials):
        data = generate_train_test(dfs, 0.80)
        traindf.append(data[0])
        testdf.append(data[1])

    # single temperature
    training = []
    testing = []
    temp_training = []
    temp_testing = []
    for it in range(num_trials):
        training.append(format_quantification(traindf[it]))
        testing.append(format_quantification(testdf[it]))
    for it in range(num_trials):
        temp_training.append(split_temps(training[it][0]))
        temp_testing.append(split_temps(testing[it][0]))

    degree_arrays = []
    for i in range(highest_degree):
        for j in range(highest_degree):
            for k in range(highest_degree):
                new_deg = [i + 1, j + 1, k + 1]
                degree_arrays.append(new_deg)

    scores = [[], [], [], []]
    min_score = Infinity
    min_tuple = ()
    for i in range(len(degree_arrays)):
        for j in range(0, 4):
            ave_score = 0.0
            for k in range(num_trials):
                score = polynomial_regression(temp_training[k][j], temp_testing[k][j], training[k][1], testing[k][1], degree_arrays[i])
                ave_score += score / num_trials
            scores[j].append(ave_score)
            if ave_score < min_score:
                min_score = ave_score
                min_tuple = degree_arrays[i]
            print (str(temps[j]) + ", DEGREES " + str(degree_arrays[i]) + " SCORE: " + str(ave_score))
    print ("MIN SCORE: " + str(min_score) + " DEGREES: " + str(min_tuple))
    