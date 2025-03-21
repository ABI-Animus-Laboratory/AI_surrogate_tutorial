'''
Created on 10/08/2020

How to execute: python run_parameter_estimation.py <surrogate_file.gen> <observation_file.csv>

@author: Gonzalo D. Maso Talou
'''

import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from models.readers import CardiacReader 

@tf.function
def train_step(domain_data, domain_labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        P_entry = tf.concat([P_ones * P_1], axis=-1)
        input_opt = tf.concat([domain_data[:,0:5], P_entry, domain_data[:,6:]], axis=-1)
        predictions_domain = surrogate_model(input_opt)
        loss = tf.reduce_sum(loss_function(domain_labels, predictions_domain))
         
    gradients = tape.gradient(loss, [P_1])
    optimizer.apply_gradients(zip(gradients, [P_1]))
     
    train_loss(loss)

if __name__ == '__main__':

    #   Load Data
    with open(sys.argv[1]) as config_file:
        cfg = json.load(config_file)

    surrogate_model = tf.keras.models.load_model(cfg["training"]["path"]+"saved_model") 
       
    print("Model restored")

    #    Read testing files
    reader = CardiacReader(cfg["network"]["inputs"], cfg["network"]["outputs"], cfg["training"]["labels"])
    input_val, output_val = reader.read_domain_values([sys.argv[2]],cfg["outputs"]["path"],[False, False, False],[0.0,0.0,0.0])
    print("Test data successfully loaded")
    
    P_1 = tf.Variable(initial_value=0.75, trainable=True, dtype=tf.float64)
    P_ones = tf.ones(shape=input_val[:,0:1].shape, dtype=tf.float64)
     
    loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1E0,decay_steps=10,decay_rate=0.985) # initial_learning_rate * decay_rate ^ (step / decay_steps)    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    history_val = []
    history_val.append([P_1.numpy()])
    tol = 1E-6
    stop = False

    #    Training
    epoch = 0
    while not stop:
        #     Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        
        train_step(input_val, output_val)
            
        print('Epoch {}, Loss: {}, P_1 : {}'.format(epoch+1, train_loss.result(), P_1.numpy()) )
         
        if np.max([np.abs(history_val[-1][0] - P_1.numpy())]) < tol:
            stop = True
        history_val.append([P_1.numpy()])
        epoch = epoch + 1

    ax = plt.plot(history_val)
    plt.title("P value")
    plt.xlabel("# epoch")
    plt.ylabel("P")
    plt.show()
    print("Converged with P = {}".format(P_1.numpy()))
