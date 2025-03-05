'''
Created on 16/09/2020

Module containing the layer classes used in BioMeS.

@author: Gonzalo D. Maso Talou
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np
import matplotlib.pyplot as plt

class Siren(Layer):
    '''
    Generates a Dense layer with sinusoidal units. The output of the layer is distributed 
    as N(0,c_sqr/6).
    '''


    def __init__(self, units=16, w_0=30.0, c_sqr=6.0, dtype="float32", is_first=False):
        '''
        Constructor
        '''
        super(Siren, self).__init__()
        self.c = np.sqrt(c_sqr)
        self.units = units
        self.w_0 = w_0
        self._dtype = dtype
        self.is_first = is_first 
        
    def build(self,input_shape):
        if self.is_first:
            w_init = tf.random_uniform_initializer(-1/np.sqrt(input_shape[-1]), 
                                                   1/np.sqrt(input_shape[-1]))
        else:
            w_init = tf.random_uniform_initializer(-self.c/np.sqrt(input_shape[-1])/self.w_0, 
                                                   self.c/np.sqrt(input_shape[-1])/self.w_0)
            
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units), dtype=self._dtype),
            trainable=True
        )
        b_init = tf.random_uniform_initializer(-1/np.sqrt(input_shape[-1]), 
                                               1/np.sqrt(input_shape[-1]))
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype=self._dtype), 
            trainable=True
        )
        
    @tf.function
    def call(self, inputs):
        inner_prod = self.w_0 * (tf.matmul(inputs, self.w) + self.b)
        output = tf.math.sin(inner_prod)
        return output#, inner_prod    


class SirenLast(Layer):
    '''
    Generates a Dense layer with linear units. 
    '''


    def __init__(self, units=16, w_0=30.0, c_sqr=6.0, dtype="float32"):
        '''
        Constructor
        '''
        super(SirenLast, self).__init__()
        self.c = np.sqrt(c_sqr)
        self.units = units
        self.w_0 = w_0
        self._dtype = dtype
        
    def build(self,input_shape):
        w_init = tf.random_uniform_initializer(-self.c/np.sqrt(input_shape[-1])/self.w_0, 
                                               self.c/np.sqrt(input_shape[-1])/self.w_0)
        
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units), dtype=self._dtype),
            trainable=True
        )
        b_init = tf.random_uniform_initializer(-1/np.sqrt(input_shape[-1]), 
                                               1/np.sqrt(input_shape[-1]))
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype=self._dtype), 
            trainable=True
        )
        
    @tf.function
    def call(self, inputs):
        output = self.w_0 * (tf.matmul(inputs, self.w) + self.b)
        return output
    
# num_samples = 1024//4
# # input = np.random.uniform(-1,1,(num_samples,1))
# values = np.linspace(-1,1,num_samples)
# input = tf.convert_to_tensor(np.reshape(values,(num_samples,1)))
# #  
# # #    --------------    Layer 1    ---------------
# layer = Siren(2048,w_0=30.0,c_sqr=6.0,dtype="float64", is_first=True)
# with tf.GradientTape(persistent=True) as tape:
#     tape.watch(input)
#     output, inter = layer(input)
# # #     inter_sum = tf.reduce_sum(inter,axis=0)
# # #     output_sum = tf.reduce_sum(output,axis=0)
# #     inter_splitted = tf.split(inter,num_or_size_splits=2048,axis=-1)
# #     output_splitted = tf.split(output,num_or_size_splits=2048,axis=-1)
# #       
# # # inter_grad = tape.gradient(inter_sum,inter)
# # # output_grad = tape.gradient(output_sum,output)
# #   
# # inter_grad = tf.convert_to_tensor([tape.gradient(current_inter,input) for current_inter in inter_splitted])
# # output_grad = tf.convert_to_tensor([tape.gradient(current_output,input) for current_output in output_splitted])
# #   
# # count, bins, ignored = plt.hist(input.numpy().flatten(), 64, density=True)
# # plt.show()
# #   
# # fig, axs = plt.subplots(2,2)
# # count, bins, ignored = axs[0,0].hist(inter.numpy().flatten(), 256, density=True)
# # count, bins, ignored = axs[1,0].hist(output.numpy().flatten(), 256, density=True)
# # count, bins, ignored = axs[0,1].hist(inter_grad.numpy().flatten(), 256, density=True)
# # count, bins, ignored = axs[1,1].hist(output_grad.numpy().flatten(), 256, density=True)
# # plt.show()
#   
# #    --------------    Layer 2    ---------------
# layer_2 = Siren(2048,dtype="float64")
# with tf.GradientTape(persistent=True) as tape:
#     tape.watch(input)
#     output, inter = layer_2(layer(input)[0])
#     inter_splitted = tf.split(inter,num_or_size_splits=2048,axis=-1)
#     output_splitted = tf.split(output,num_or_size_splits=2048,axis=-1)
#        
# inter_grad = tf.convert_to_tensor([tape.gradient(current_inter,input) for current_inter in inter_splitted])
# output_grad = tf.convert_to_tensor([tape.gradient(current_output,input) for current_output in output_splitted])
#    
# fig, axs = plt.subplots(2,2)
# count, bins, ignored = axs[0,0].hist(inter.numpy().flatten(), 256, density=True)
# count, bins, ignored = axs[1,0].hist(output.numpy().flatten(), 256, density=True)
# count, bins, ignored = axs[0,1].hist(inter_grad.numpy().flatten(), 256, density=True)
# count, bins, ignored = axs[1,1].hist(output_grad.numpy().flatten(), 256, density=True)
# plt.show()

