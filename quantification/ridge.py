import pandas as pd
import numpy as np
import seaborn as sns
from data_io import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def linear_regression(temp_train, temp_test, target_train, target_test):
    # initializes and trains the model
    model = Ridge()
    model.fit(temp_train, target_train)

    # tests the trained model and find accuracy of the model
    predicted = model.predict(temp_test)
    score_arr = percent_error(target_test, predicted, 1.0)

    # returns the percent error of the model
    return ave_arr(score_arr)

if __name__ == '__main__':
    # finds algorithm name and makes a folder to hold images if the folder does not already exist
    path_name = sys.argv[0][:-3]
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG+NH3"]
    num_trials = 100

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

    scores = []
    for j in range(0, 4):
        ave_score = 0.0
        for k in range(num_trials):
            score = linear_regression(temp_training[k][j], temp_testing[k][j], training[k][1], testing[k][1])
            ave_score += score / num_trials
        scores.append(ave_score)
        print (str(temps[j]) + " SCORE: " + str(ave_score))
    print ("NUMBER OF TRIALS: " + str(num_trials))