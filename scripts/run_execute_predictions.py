'''
Created on 18/03/2021

@author: Gonzalo D. Maso Talou
'''

import sys
import json
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from models.readers import CardiacReader 

if __name__ == '__main__':

    with open(sys.argv[1]) as config_file:
        cfg = json.load(config_file)

    #   Load Model
    model = tf.keras.models.load_model(cfg["training"]["path"]+"saved_model")    
    print("Model restored")

    #    Read input prediction file/s
    reader = CardiacReader(cfg["network"]["inputs"], cfg["network"]["outputs"], cfg["training"]["labels"])
    input_val, output_val = reader.read_and_filtered_data([sys.argv[2]],cfg["outputs"]["path"],["P"],[0.9])
    print("Input data successfully loaded")
    
    prediction = model(input_val)

    x_val = input_val[:,0]
    y_val = input_val[:,1]
    z_val = input_val[:,2]
    dx_val = output_val[:,0]
    dy_val = output_val[:,1]
    dz_val = output_val[:,2]

    dx_pred = prediction[:,0]
    dy_pred = prediction[:,1]
    dz_pred = prediction[:,2]

    #    Some I/O examples to manipulate model outputs
    error = ( np.sum(np.sqrt((dx_val-dx_pred)**2 + (dy_val-dy_pred)**2 + (dz_val-dz_pred)**2)) ) / len(dx_val)
    max_displacement = np.amax( np.sqrt( (dx_val)**2 + (dy_val)**2 + (dz_val)**2 ) )
    mean_displacement = np.mean( np.sqrt( (dx_val)**2 + (dy_val)**2 + (dz_val)**2 ) )
    print("Mean absolute global error {}, mean displacement {}, max displacement {}".format(error, mean_displacement, max_displacement) )
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x_val, y_val, z_val*1.1, dx_val, dy_val, dz_val, length=1.0, normalize=False, color='red', label='Ground truth')
    ax.quiver(x_val, y_val, z_val, dx_pred, dy_pred, dz_pred, length=1.0, normalize=False, color='blue', label='Prediction')
    ax.set_ylabel(r'$x$')
    ax.set_xlabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x_val, y_val, z_val, dx_pred, dy_pred, dz_pred, length=1.0, normalize=False, color='blue',
              label='Prediction')
    ax.set_ylabel(r'$x$')
    ax.set_xlabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x_val, y_val, z_val, dx_val-dx_pred, dy_val-dy_pred, dz_val-dz_pred, length=1.0, normalize=False, color='red',
              label='Difference GT w.r.t. prediction')
    ax.set_ylabel(r'$x$')
    ax.set_xlabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.legend()
    plt.show()
