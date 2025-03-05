'''
Created on 4/08/2020

@author: Gonzalo D. Maso Talou
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.feature_column.feature_column import input_layer
        
class DenseSurrogate(Model):
    '''
    Surrogate with a basic MLP architecture.
    '''

    def __init__(self, num_inputs, num_outputs, MLP_layers, activation = tf.nn.relu ):
        '''
        Constructor
        '''
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.MLP_layers = MLP_layers
        
        self.BN = []
        self.MLP = []
        
        for idx_output in range(0,self.num_outputs):
            self.BN.append([])
            self.MLP.append([])

        for idx_output in range(0,self.num_outputs):
            for idx_layer in range(0,len(self.MLP_layers)):
                self.BN[idx_output].append( tf.keras.layers.BatchNormalization() )
                self.MLP[idx_output].append( tf.keras.layers.Dense(self.MLP_layers[idx_layer], activation = activation, kernel_initializer='he_normal') )
            #    Output layer is always a scalar
            self.MLP[idx_output].append( tf.keras.layers.Dense(1) )

#     @tf.function    
    def call(self,inputs):

        #    Initialise the input of the N MLP networks
        outputs = []
        for idx_output in range(0,self.num_outputs):
            outputs.append(inputs)
            
        #    Construct the graph for the N MLP networks
        for idx_output in range(0,self.num_outputs):
            for idx_layer in range(0,len(self.MLP_layers)):
                outputs[idx_output] = self.BN[idx_output][idx_layer](outputs[idx_output])
                outputs[idx_output] = self.MLP[idx_output][idx_layer](outputs[idx_output])
            outputs[idx_output] = self.MLP[idx_output][-1](outputs[idx_output])
     
        outputs = tf.concat(outputs, axis = -1)
                
        return outputs

    def initialise(self,dataset_training):
        for batch_domain in dataset_training:
            self._set_inputs(batch_domain[0])
            break
        
