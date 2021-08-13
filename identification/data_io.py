import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sets predetermined ranges and corresponding labels dividing different 
# experiment conditions (sets of constituent gasses) as well as temperatures
labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG + NH3"]
temperatures = [450, 500, 550, 600]
ranges = [(0, 12), (48, 115), (35, 47), (24, 34), (179, 211)]
col_names = ['V0_450', 'V0_500', 'V0_550', 'V0_600', 'V1_450', 'V1_500', 'V1_550', 'V1_600', 'V2_450', 'V2_500', 'V2_550', 'V2_600']

''' 
    order of labels, ranges (inclusive):
    0 - CH4 only    (0, 12)
    1 - CH4 + NH3   (48, 115)
    2 - NG only     (35, 47)
    3 - NH3 only    (24, 34)
    4 - NG + NH3    (179, 211)

    0-2 are for the 3 label case
    0-4 are for the 5 label case
'''

def average_mat(mat_arr):
    # finds the nxn dimension of the matrix and initializes an empty matrix 
    dim = len(mat_arr[0])
    ave_mat = []
    for i in range(dim):
        ave_mat.append([])
        for j in range(dim):
            ave_mat[i].append(0.0)

    # iterates through each matrix and adds the normalized value at each entry to the empty matrix
    for mat in mat_arr:
        for row in range(len(mat)):
            for col in range(len(mat[row])):
                ave_mat[row][col] += mat[row][col] / len(mat_arr) * 100

    # returns an averaged matrix
    return ave_mat

def save_graph(alg_name, temp_arr, conf_mat):
    # initializes confusion matrix frame to fill with labels
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10,10))
    target_names = labels[0:len(conf_mat)]

    # graphs the confusion matrix as a heat map
    sns.heatmap(conf_mat, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap='Blues_r')
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')

    # determines path name and file name of the figure
    temp_str = ''
    for temp in temp_arr:
        temp_str += (str(temp).zfill(3) + '_')
    prog_name = alg_name.split('/')[0]
    path_name = alg_name + '/' + prog_name + '_' + str(len(conf_mat)) + 'label_' + temp_str
    
    # saves the figure
    plt.savefig(path_name[:-1] + '.png')
    plt.close()

def save_eval_graph(alg_name, temp_arr, conf_mat, estimators):
    # initializes confusion matrix frame to fill with labels
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10,10))
    target_names = labels[0:len(conf_mat)]

    # graphs the confusion matrix as a heat map
    sns.heatmap(conf_mat, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap='Blues_r')
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')
    plt.title(str(estimators) + " estimators")

    # determines path name and file name of the figure
    temp_str = ''
    for temp in temp_arr:
        temp_str += (str(temp).zfill(4) + '_')
    prog_name = alg_name.split('/')[0]
    path_name = alg_name + '/' + prog_name + '_' + str(len(conf_mat)) + 'label_' + temp_str
    
    # saves the figure
    plt.savefig(path_name[:-1] + '.png')
    plt.close()

def concat_data(arr1, arr2):
    # input checking
    if len(arr1) == 0:
        return arr2
    if len(arr2) == 0:
        return arr1

    # initializes a new array to add concatanated data to
    concat = []

    # iterates through each row of data and concatanates data
    for i in range(0, len(arr1)):
        new_arr = []
        for a in arr1:
            new_arr.append(a)
        for a in arr2:
            new_arr.append(a)
        concat.append(a)

    # returns an array of concatanated data
    return concat


def split_temps(data):
    # initializes temperature data arrays to fill and return
    temp_arr = [[], [], [], []]

    # iterates through dataset and separates each temperature
    for row in data:
        for temp in range(0, 4):
            new_set = []
            for sensor in range(0, 3):
                new_set.append(row[temp + (sensor * 4)])
            temp_arr[temp].append(new_set)

    # returns array of arrays, each holding the data for a given temperature
    return temp_arr


def format_identification(df_arr, num_labels):
    # sets empty arrays for formatted data and assigned labels
    formatted = []
    target_vals = []

    # iterates through each constituent gas group data, and line of data in the gas groups, formatting the data to an easily readable format
    for label in range(0, num_labels):
        for index, row in (df_arr[label]).iterrows():
            new_row = []
            for name in col_names:
                new_row.append(row[name])
            target_vals.append(label)
            formatted.append(new_row)
        
    # returns formatted data as a pandas dataframe
    return [formatted, target_vals]


def generate_train_test(df_arr, batch_prop):
    training = []
    testing = []

    # loops through each gas group in the data
    for i in range(0, len(df_arr)):
        single_df = df_arr[i]

        # pulls given batch proportion size for the training dataframe
        sample_train_df = single_df.sample(frac=batch_prop)

        # creates a new, empty data frame to add testing data
        sample_test_df = pd.DataFrame(columns = ['Unnamed: 0', 'CH4/ppm', 'H2/ppm', 'NH3/ppm', 'NatGas_CH4/ppm', 'V0_500', 'V0_550', 'V0_600', 'V1_500', 'V1_550', 'V1_600', 'V2_500', 'V2_550', 'V2_600', 'V0_450', 'V1_450', 'V2_450'])

        # iterates through dataframe and adds remaining data to testing data
        train_data = sample_train_df.iloc[:,0].values
        for index, row in single_df.iterrows():
            if row['Unnamed: 0'] not in train_data:
                sample_test_df = sample_test_df.append(row, ignore_index=True)

        # adds testing and training data to final dataframe array
        training.append(sample_train_df)
        testing.append(sample_test_df)

        # sample length verification, uncomment to verify
        '''
        print ("Group " + str(i))
        print ("\ttot len "+ str(i) + ":\t" + str(len(df_arr[i])))
        print ("\ttrain len "+ str(i) + ":\t" + str(len(sample_train_df)))
        print ("\ttest len "+ str(i) + ":\t" + str(len(sample_test_df)))
        '''

    # returns two arrays of training and testing dataframes, respectively
    # each array is ordered in constituent gas groups
    return training, testing


def load_data():
    # import data into pandas data frame
    df = pd.read_csv('../202103_Gen4Sensor_out.csv')

    # loads each experimental condition into its own dataframe
    subdf = []
    for bounds in ranges:
        new_df = df[((df['Unnamed: 0'] <= bounds[1]) & (df['Unnamed: 0'] >= bounds[0]))]
        subdf.append(new_df)

    # returns array of dataframes
    return subdf