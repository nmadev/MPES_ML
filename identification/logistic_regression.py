import pandas as pd
import numpy as np
import seaborn as sns
from data_io import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

def lin_reg(temp_train, temp_test, target_train, target_test):
    # initializes and trains the model
    model = LogisticRegression(class_weight = 'balanced', multi_class='ovr')
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
    path_two = path_name + '/' + 'pair'
    path_three = path_name + '/' + 'trio'
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if not os.path.exists(path_one):
        os.makedirs(path_one)
    if not os.path.exists(path_two):
        os.makedirs(path_two)
    if not os.path.exists(path_three):
        os.makedirs(path_three)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG+NH3"]

    # import data into pandas data frame
    dfs = load_data()

    # randomly samples 80% of the data for training and 20% for testing
    traindf = []
    testdf = []
    for it in range(3):
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
        for it in range(3):
            training.append(format_identification(traindf[it], n_label))
            testing.append(format_identification(testdf[it], n_label))
        for it in range(3):
            temp_training.append(split_temps(training[it][0]))
            temp_testing.append(split_temps(testing[it][0]))

        for j in range(0, 4):
            mat_arr = []
            for k in range(0, 3):
                conf_mat = lin_reg(temp_training[k][j], temp_testing[k][j], training[k][1], testing[k][1])
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                mat_arr.append(np.transpose(conf_mat))
            ave_mat = average_mat(mat_arr)

            save_graph(path_one, [temps[j]], ave_mat)

        # pair-wise temperature
        for j in range(0, 16):
            a = j % 4
            b = j // 4
            if a >= b:
                continue
            pair_train = []
            pair_test = []
            for it in range(3):
                pair_train.append(concat_data(temp_training[it][a], temp_training[it][b]))
                pair_test.append(concat_data(temp_testing[it][a], temp_testing[it][b]))

            for k in range(0, 3):
                conf_mat = lin_reg(pair_train[k], pair_test[k], training[k][1], testing[k][1])
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                mat_arr.append(np.transpose(conf_mat))
            ave_mat = average_mat(mat_arr)

            save_graph(path_two, [temps[a], temps[b]], ave_mat)

        # three temperature
        for j in range(0, 4):
            temp_indices = []
            trio_train = [[], [], []]
            trio_test = [[], [], []]

            for it in range(0, 3):
                for k in range(0, 4):
                    if j == k:
                        continue
                    if it == 0:
                        temp_indices.append(k)
                    trio_train[it] = concat_data(trio_train[it], temp_training[it][k])
                    trio_test[it] = concat_data(trio_test[it], temp_testing[it][k])
            
            for k in range(0, 3):
                conf_mat = lin_reg(trio_train[k], trio_test[k], training[k][1], testing[k][1])
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                mat_arr.append(np.transpose(conf_mat))
            ave_mat = average_mat(mat_arr)

            temps_arr = []
            for index in temp_indices:
                temps_arr.append(temps[index])
            save_graph(path_three, temps_arr, ave_mat)
            
        # four temperature
        quad_train = []
        quad_test = []

        for it in range(0, 3):
            train_temporary = concat_data(concat_data(temp_training[it][0], temp_training[it][1]), concat_data(temp_training[it][2], temp_training[it][3]))
            test_temporary = concat_data(concat_data(temp_testing[it][0], temp_testing[it][1]), concat_data(temp_testing[it][2], temp_testing[it][3]))
            quad_train.append(train_temporary)
            quad_test.append(test_temporary)

        for k in range(0, 3):
            conf_mat = lin_reg(quad_train[k], quad_test[k], training[k][1], testing[k][1])
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            mat_arr.append(np.transpose(conf_mat))
        ave_mat = average_mat(mat_arr)

        save_graph(path_name, temps, ave_mat)