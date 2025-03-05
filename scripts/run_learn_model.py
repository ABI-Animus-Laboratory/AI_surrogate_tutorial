'''
Created on 04/08/2020
Modified on 05/03/2025

@summary: Training script of an AI-surrogate model using an MLP architecture and BC constraints.

@author: Gonzalo D. Maso Talou
'''

import os, sys, json
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np
import math
from time import time

from models.readers import CardiacReader 
from models.surrogates import DenseSurrogate
from models.generators import DualStiffnessBCGenerator, SingleStiffnessBCGenerator
    
@tf.function
def train_step(domain_data, domain_labels, BN_data, BN_labels, BD_data, BD_labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions_domain = surrogate_model(domain_data)
        predictions_BD = surrogate_model(BD_data)
        predictions_BN = surrogate_model(BN_data)
        loss_domain = tf.reduce_sum(loss_function(domain_labels, predictions_domain)) / batch_size
        loss_BN = tf.reduce_sum(loss_function(BN_labels, predictions_BN)) / batch_size
        loss_BD = tf.reduce_sum(loss_function(BD_labels, predictions_BD)) / batch_size
        loss = weights[0] * loss_domain + weights[1] * loss_BN + weights[2] * loss_BD
         
    gradients = tape.gradient(loss, surrogate_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, surrogate_model.trainable_variables))
     
    train_loss(loss)
    domain_error(loss_domain)
    BN_error(loss_BN)
    BD_error(loss_BD)
  
@tf.function
def validation_step(data, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = surrogate_model(data, training=False)
    t_loss = tf.reduce_sum(loss_function(labels, predictions)) / batch_size
    
    validation_loss(t_loss)

def initialise_metrics():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    domain_error = tf.keras.metrics.Mean(name='domain_error')
    BN_error = tf.keras.metrics.Mean(name='BN_error')
    BD_error = tf.keras.metrics.Mean(name='BD_error')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    best_metric = float('inf')
    return train_loss, validation_loss, domain_error, BN_error, BD_error, best_metric

def reset_metrics(train_loss, validation_loss, domain_error, BN_error, BD_error):
    train_loss.reset_states()
    validation_loss.reset_states()
    domain_error.reset_states()
    BN_error.reset_states()
    BD_error.reset_states()

def split_training_testing(files, test_ratio):
    NUM_FILES = len(files)
    testing_idx = np.random.choice(NUM_FILES, int(test_ratio*NUM_FILES), replace=False)
    training_idx = np.delete(np.arange(NUM_FILES),testing_idx)
    train_files = [files[i] for i in training_idx]
    test_files = [files[i] for i in testing_idx]
    return train_files, test_files

if __name__ == '__main__':

    tf.keras.backend.set_floatx('float32')
    
    #   Load Data
    with open(sys.argv[1]) as config_file:
        cfg = json.load(config_file)

    input_path = cfg["outputs"]["path"]
    output_path = cfg["training"]["path"]
    batch_size = int(cfg["training"]["batch_size"])
    max_epoch = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    decay_rate = cfg["training"]["decay_per_epoch"]
    weights = cfg["training"]["weights"]
    condition_BN = cfg["training"]["condition_BN"]  #    Neumann boundary condition
    values_BN = cfg["training"]["values_BN"]        #    Neumann boundary condition
    condition_BD = cfg["training"]["condition_BD"]  #    Dirichlet boundary condition
    values_BD = cfg["training"]["values_BD"]        #    Dirichlet boundary condition
    labels = cfg["training"]["labels"]

    #    Loading CSV files containing samples
    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    train_files, test_files = split_training_testing(onlyfiles, cfg["training"]["testing_ratio"])

    #    Loading of configuration parameters for the network architecture
    neurons_per_layers_ann = cfg["network"]["neurons_ann"]#[64, 64, 64, 1]
    num_inputs = cfg["network"]["inputs"]
    num_outputs = cfg["network"]["outputs"]
    file_id = cfg["training"]["training_ID"]+"_"+str(neurons_per_layers_ann[0])+"x"+str(len(neurons_per_layers_ann)-1) +"_Batch"+str(batch_size)+"_lr"+str(learning_rate)+"_epochs"+str(max_epoch)

    #    Generation of a log file
    logfile = output_path + file_id + ".log"
    f = open(logfile,"w+")
    
    #    Loading of the dataset
    reader = CardiacReader(num_inputs, num_outputs, labels)
    f.write("Loading for training {}\n".format(train_files))
    input_train, output_train = reader.read_domain_values(train_files,input_path,condition_BN,values_BN)
    input_train_BN, output_train_BN = reader.read_neumann_boundary(train_files,input_path,condition_BN,values_BN)
    f.write("Loading for validation {}\n".format(test_files))
    input_val, output_val = reader.read_domain_values(test_files,input_path,[False, False, False],[0.0,0.0,0.0])
    f.flush()

    num_training_samples = input_train.shape[0]
    num_validation_samples = input_val.shape[0]
    shuffle = min(num_training_samples,100000)

    #    Training from here on
    #    Datasets constructors
    print("Generating dataset_training")
    dataset_training = tf.data.Dataset.from_tensor_slices((input_train, output_train))
    dataset_training = dataset_training.shuffle(shuffle).batch(batch_size).prefetch(1)
 
    dataset_validation = tf.data.Dataset.from_tensor_slices((input_val, output_val))
    dataset_validation = dataset_validation.batch(batch_size)
     
    dataset_BN = tf.data.Dataset.from_tensor_slices((input_train_BN, output_train_BN))
    dataset_BN = dataset_BN.shuffle(shuffle).repeat().batch(batch_size)
    
    #    Generator
    generator = DualStiffnessBCGenerator()    
    dataset_BD = tf.data.Dataset.from_generator(generator=lambda: generator.generate(batch_size,condition_BD,values_BD), 
                                                output_types=(tf.float64, tf.float64),
                                                output_shapes=([None,num_inputs], [None,num_outputs]) )
    dataset_BD = dataset_BD.repeat()

    #    Surrogate NN
    surrogate_model = DenseSurrogate(num_inputs,num_outputs,MLP_layers=neurons_per_layers_ann[0:-1])
    surrogate_model.initialise(dataset_training)

    #    Tensorboard initialisation
    tensorboard_log = tf.keras.callbacks.TensorBoard(output_path)
    tensorboard_log.set_model(surrogate_model)    

    #    Loss and numerical scheme
    loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,decay_steps=math.ceil(num_training_samples/batch_size),
                                                                 decay_rate=decay_rate) # initial_learning_rate * decay_rate ^ (step / decay_steps)    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    #    Metrics
    train_loss, validation_loss, domain_error, BN_error, BD_error, best_metric = initialise_metrics()

    #    Training
    for epoch in range(cfg["training"]["epochs"]):
        start = time()

        #     Reset the metrics at the start of the next epoch
        reset_metrics(train_loss, validation_loss, domain_error, BN_error, BD_error)
        
        idx=1 
        #    Training loop
        for batch_domain, batch_BN, batch_BD in zip(dataset_training,dataset_BN,dataset_BD):
            print("Batch {}/{}".format(idx,math.ceil(num_training_samples/batch_size)), end='\r')
            idx = idx + 1
            train_step(batch_domain[0],batch_domain[1],batch_BN[0],batch_BN[1],batch_BD[0],batch_BD[1])
            
        idx=1 
        #    Validation loop
        for batch in dataset_validation:
            print("Batch {}/{}".format(idx,math.ceil(num_validation_samples/batch_size)), end='\r')
            idx = idx + 1
            validation_step( batch[0], batch[1] )

        end = time()

        template = 'Epoch {}/{}, Loss: {}, Domain error: {}, BN error: {}, BD error: {}, Validation Loss: {}, Exec time: {}'
        print(template.format(epoch+1,cfg["training"]["epochs"],
                              train_loss.result(), domain_error.result(), BN_error.result(), BD_error.result(), 
                              validation_loss.result(), end-start ))
         
        #    If the current set of weights perform better in the validation set, update the weights of the surrogate model candidate
        current_metric = validation_loss.result()
        if current_metric < best_metric:
            surrogate_model.save(output_path+"saved_model", save_format="tf")    
            best_metric = current_metric

        #    TensorBoard
        logs = {'Loss': train_loss.result(), 'Domain Error': domain_error.result(), 'Neumann BD Error': BN_error.result(), 
                'Dirichlet BD Error': BD_error.result(), 'Validation Loss': validation_loss.result()}
        tensorboard_log.on_epoch_end(epoch, logs)
          
    tensorboard_log.on_train_end('_')  

    #    Save the best surrogate model candidate
    surrogate_model.save(output_path+"saved_model_final", save_format="tf")    
