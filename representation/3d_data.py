from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_io import *
import scipy as scp
import sys
import os

if __name__ == '__main__':
    # declaration of some constants
    temps = [450, 500, 550, 600]
    color_arr = [(1.0, 0.0, 0.0, 1.0),
                 (0.0, 1.0, 0.0, 1.0),
                 (0.0, 0.0, 1.0, 1.0),
                 (1.0, 0.0, 1.0, 1.0),
                 (0.0, 1.0, 1.0, 1.0)]
    labels = ["CH4 Only", "CH4 + NH3", "NG Only", "NH3 Only", "NG + NH3"]
    col_names = ['V0_450', 'V0_500', 'V0_550', 'V0_600', 'V1_450', 'V1_500', 'V1_550', 'V1_600', 'V2_450', 'V2_500', 'V2_550', 'V2_600']

    # normalization temperature, 0-3 --> 450-600
    temp_normal = 0
    # graphs number of labels in order of array labels
    num_labels = 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # import data into pandas data frame
    dfs = load_data()
    x = []
    y = []
    z = []

    for df in dfs:
        x.append(np.asarray(df[col_names[temp_normal + 0]]))
        y.append(np.asarray(df[col_names[temp_normal + 4]]))
        z.append(np.asarray(df[col_names[temp_normal + 8]]))

    for i in range(num_labels):
        ax.scatter(x[i], y[i], z[i], color=color_arr[i], label=labels[i])

    leg = ax.legend()
    ax.set_xlabel('ITO vs. Pt. (Volts)')
    ax.set_ylabel('LSC vs. Pt (Volts)')
    ax.set_zlabel('Au vs. Pt (Volts)')
    plt.show()