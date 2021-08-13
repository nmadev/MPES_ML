import pandas as pd
import numpy as np
import seaborn as sns
from data_io import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

def naive_bayes(temp_train, temp_test, target_train, target_test):
    # initializes and trains the model
    model = GaussianNB()
    model.fit(temp_train, target_train)

    # tests the trained model and creates a confusion matrix
    predicted = model.predict(temp_test)
    conf_mat = confusion_matrix(target_test, predicted)

    # returns the confusion matrix from testing
    return conf_mat

if __name__ == '__main__':
    # finds algorithm name and makes a folder to hold images if the folder does not already exist
    path_name = sys.argv[0][:-3]
    path_one = path_name + '/' + 'single'
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if not os.path.exists(path_one):
        os.makedirs(path_one)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG+NH3"]
    num_trials = 1000
    
    # import data into pandas data frame
    dfs = load_data()

    # randomly samples 80% of the data for training and 20% for testing
    traindf = []
    testdf = []
    for it in range(num_trials):
        data = generate_train_test(dfs, 0.80)
        traindf.append(data[0])
        testdf.append(data[1])

    for i in [3, 5]:
        n_label = i

        # single temperature
        training = []
        testing = []
        temp_training = []
        temp_testing = []
        for it in range(num_trials):
            training.append(format_identification(traindf[it], n_label))
            testing.append(format_identification(testdf[it], n_label))
        for it in range(num_trials):
            temp_training.append(split_temps(training[it][0]))
            temp_testing.append(split_temps(testing[it][0]))

        for j in range(0, 4):
            mat_arr = []
            for k in range(num_trials):
                # train and test the model
                conf_mat = naive_bayes(temp_training[k][j], temp_testing[k][j], training[k][1], testing[k][1])
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                mat_arr.append(np.transpose(conf_mat))
            ave_mat = average_mat(mat_arr)

            save_graph(path_one, [temps[j]], ave_mat)