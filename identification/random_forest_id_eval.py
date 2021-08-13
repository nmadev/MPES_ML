import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_io import *
import scipy as scp
import sklearn
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

# global declarations
num_trials = 100
def random_forest(temp_train, temp_test, target_train, target_test, estimators_num):
    # initializes and trains the model
    model = RandomForestClassifier(n_estimators=estimators_num)
    model.fit(temp_train, target_train)

    # tests the trained model and creates a confusion matrix
    predicted = model.predict(temp_test)
    conf_mat = confusion_matrix(target_test, predicted)
    score = accuracy_score(target_test, predicted)

    # returns the confusion matrix from testing
    return score, conf_mat

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
    estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    scores_3 = []
    scores_5 = []

    # import data into pandas data frame
    dfs = load_data()

    # randomly samples 80% of the data for training and 20% for testing
    traindf = []
    testdf = []
    for it in range(num_trials):
        data = generate_train_test(dfs, 0.80)
        traindf.append(data[0])
        testdf.append(data[1])
    
    max_3 = 0.0
    max_5 = 0.0

    for estimator in estimators:
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

            mat_arr = []
            ave_score = 0.0
            for k in range(num_trials):
                new_score, conf_mat = random_forest(temp_training[k][0], temp_testing[k][0], training[k][1], testing[k][1], estimator)
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                mat_arr.append(np.transpose(conf_mat))
                ave_score += new_score / num_trials
            ave_mat = average_mat(mat_arr)

            if i == 3:
                scores_3.append(ave_score)
            else:
                scores_5.append(ave_score)

            save_eval_graph(path_one, [temps[0], estimator], ave_mat, estimator)
            
            # prints current evaluation status
            print ("Finished " + str(i) + "-label " + str(estimator) + " estimators: accuracy " + str(ave_score * 100))

    df = pd.DataFrame({'estimators': estimators, '3 label': scores_3, '5 label': scores_5})
        
    # graphs accuracy vs. estimators for the 3-label case
    sns.set_theme(style="darkgrid")
    splot = sns.lineplot(x="estimators", y="3 label", data=df)
    splot.set(xscale="log", xlabel="Number of Estimators", ylabel="Accuracy", title="Accuracy vs. Number of Estimators for 3 Labels")
    bottom, top = splot.get_ylim()
    plt.savefig(path_name + "/3_label_estimators.png")
    plt.close()

        
    # graphs accuracy vs. estimators for the 5-label case
    sns.set_theme(style="darkgrid")
    splot = sns.lineplot(x="estimators", y="5 label", data=df)
    splot.set(xscale="log", xlabel="Number of Estimators", ylabel="Accuracy", title="Accuracy vs. Number of Estimators for 5 Labels")
    plt.savefig(path_name + "/5_label_estimators.png")
    plt.close()

    # saves the 3-label and 5-label accuracy data to a csv file
    data_out = pd.DataFrame(estimators, columns = ['estimators'])
    data_out['3-label'] = pd.Series(scores_3, index=data_out.index)
    data_out['5-label'] = pd.Series(scores_5, index=data_out.index)
    data_out.to_csv(path_name + '/accuracy_data.csv', encoding='utf-8', index=True)