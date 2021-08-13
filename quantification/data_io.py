from numpy.core.numeric import Infinity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sets predetermined ranges and corresponding labels dividing different 
# experiment conditions (sets of constituent gasses) as well as temperatures
labels = ["CH4", "CH4 + NH3", "NG", "NH3", "NG + NH3"]
quant_gas = ['CH4/ppm', 'NatGas_CH4/ppm']
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
def ave_arr(arr):
    # calculates the average of an array of numbers
    return sum(arr) / len(arr) * 100

def max_arr(arr):
    # returns the maximum value of an array
    max_val = 0.0
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

def min_arr(arr):
    # returns the minimum value of an array
    min_val = Infinity
    for i in range(len(arr)):
        if arr[i] < min_val:
            min_val = arr[i]
    return min_val

def percent_error(actual, predicted, smoothing):
    # calculates the percent error between the actual and predicted values
    error_arr = []
    for i in range(len(predicted)):
        if actual[i] == 0:
            actual[i] += smoothing
        error = abs(predicted[i] - actual[i]) / actual[i]
        error_arr.append(error)
    return error_arr

def percent_error_bars(actual, predicted):
    # calculates the max and min percent error for the given data
    maximum = 0.0
    minimum = Infinity
    for i in range(len(predicted)):
        error = abs(predicted[i] - actual[i]) / actual[i]
        if error > maximum:
            maximum = error
        elif error < minimum:
            minimum = error
    return (minimum * 100, maximum * 100)

def concat_data(arr1, arr2):
    # concatenates the two data arrays into one
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
    # splits the data into different dataframes for each temperature value
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


def format_quantification(df_arr):
    # formats the quantification data for each gas group for use in the regression
    # sets empty arrays for formatted data and assigned labels
    formatted = []
    target_vals = []

    # iterates through each constituent gas group data, and line of data in the gas groups, formatting the data to an easily readable format
    for label in range(0, 3):
        for index, row in (df_arr[label]).iterrows():
            new_row = []
            new_target = 0.0
            for name in col_names:
                new_row.append(row[name])
            for name in quant_gas:
                new_target += float(row[name])
            target_vals.append(new_target)
            formatted.append(new_row)
        
    # returns formatted data as a pandas dataframe
    return [formatted, target_vals]


def generate_train_test(df_arr, batch_prop):
    # generates a random train and test set for each dataframe in the array
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

    # returns two arrays of training and testing dataframes, respectively
    # each array is ordered in constituent gas groups
    return training, testing


def load_data():
    # loads the data from the csv file, formats it, and returns it as a pandas dataframe
    # import data into pandas data frame
    df = pd.read_csv('../202103_Gen4Sensor_out.csv')

    # loads each experimental condition into its own dataframe
    subdf = []
    for bounds in ranges:
        new_df = df[((df['Unnamed: 0'] <= bounds[1]) & (df['Unnamed: 0'] >= bounds[0]))]
        subdf.append(new_df)

    # returns array of dataframes
    return subdf