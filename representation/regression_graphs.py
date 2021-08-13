from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_io import *
import scipy as scp
import sys
import os

def split_arr(arr):
    new_arr = [[], [], []]
    split_ranges = [(0, 12), (13, 80), (81, 93)]
    for i in range(len(arr)):
        for j in range(len(split_ranges)):
            if i >= split_ranges[j][0] and i <= split_ranges[j][1]:
                new_arr[j].append(arr[i])
    return new_arr

if __name__ == '__main__':
    # make folder to hold graphs
    path_name = sys.argv[0][:-3]
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # declaration of some constants
    temps = [450, 500, 550, 600]
    color_arr = ['green', 'red', 'blue']
    labels = ["CH4 Only", "CH4 + NH3", "NG Only", "NH3 Only", "NG + NH3"]
    sensors = ["ITO", "LSC", "Au"]
    col_names = ['V0_450', 'V0_500', 'V0_550', 'V0_600', 'V1_450', 'V1_500', 'V1_550', 'V1_600', 'V2_450', 'V2_500', 'V2_550', 'V2_600']


    # import data into pandas data frame
    dfs = load_data()
    quant = format_quantification(dfs)
    rot_quant = rotate_matrix(np.asarray(quant[0]))
    temp_data = split_temps(rot_quant)
    temp_solution = split_arr(np.asarray((quant[1])))
    # DATA: rot_quant[i]
    # SOLUTION: quant[1]
    for temp in range (4):
        for sensor in range(3):
            dep = split_arr(rot_quant[temp + 4 * sensor])
            for i in range(3):
                plt.scatter(temp_solution[i], dep[i], c=color_arr[i], label=labels[i])
            plt.title(sensors[sensor] + "/Pt Sensor, " + str(temps[temp]) + "C")
            plt.xlabel("Methane Concentration (PPM)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.savefig(path_name + "/" + str(temps[temp]) + "C_" + sensors[sensor] + ".png")
            plt.close()