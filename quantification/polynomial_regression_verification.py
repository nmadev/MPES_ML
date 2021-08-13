from matplotlib.pyplot import yscale
from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np
import seaborn as sns
from data_io import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_regression(temp_train, temp_test, target_train, target_test, n):
    model = Pipeline([('poly', PolynomialFeatures(degree=n)), ('linear', LinearRegression(fit_intercept=False))])
    # initializes and trains the model
    model.fit(temp_train, target_train)

    # tests the trained model and finds accuracy of the model
    predicted = model.predict(temp_test)
    score_arr = percent_error(target_test, predicted, 1.0)
    error_bounds = percent_error_bars(target_test, predicted)
    print ("target_test")
    print (target_test)
    print ("predicted")
    print (predicted)
    
    # returns the percent error of the model and max/min percent error
    return ave_arr(score_arr), error_bounds

if __name__ == '__main__':
    # finds algorithm name and makes a folder to hold images if the folder does not already exist
    path_name = sys.argv[0][:-3]
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG+NH3"]
    num_trials = 1
    highest_degree = 10

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

    scores = [[], [], [], []]
    min_err = [[], [], [], []]
    max_err = [[], [], [], []]
    for i in [2]:
        for j in [0]:
            ave_score = 0.0
            ave_min = 0.0
            ave_max = 0.0
            for k in range(num_trials):
                score = polynomial_regression(temp_training[k][j], temp_testing[k][j], training[k][1], testing[k][1], i + 1)
                ave_score += score[0] / num_trials
                ave_min += score[1][0] / num_trials
                ave_max += score[1][1] / num_trials
            scores[j].append(ave_score)
            min_err[j].append(ave_min)
            max_err[j].append(ave_max)
            print (str(temps[j]) + ", DEGREE " + str(i + 1) + " SCORE: " + str(ave_score))
            print (str(ave_min), str(ave_max))
    print ("NUMBER OF TRIALS: " + str(num_trials))

    # plots the scores for each temperature
    for i in range(0, 4):
        df = pd.DataFrame()
        df['degree'] = range(1, highest_degree + 1)
        df['percent_error'] = scores[i]
        df['max_err'] = max_err[i]
        df['min_err'] = min_err[i]
        sns.scatterplot(data=df, x='degree', y='percent_error')
        plt.title('Percent Error vs. Polynomial Degree for ' + str(temps[i]) + 'C')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Percent Error')
        plot_name = str(temps[i]) + "_" + path_name + ".png"
        plt.savefig(path_name + "/" + plot_name)
        plt.close()