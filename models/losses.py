'''
Created on 02/10/2020

@author: Gonzalo D. Maso Talou
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss

class MeanDistanceToSet(Loss):
    '''
    Mean euclidean distance of the predictions to a set of label points. For each prediction,
    the minimum distance label is chosen for comparison. The y set with more elements should 
    go last to ensure a more meaningful comparison.
    '''
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO ):
        '''
        Constructor
        '''
        super(MeanDistanceToSet, self).__init__(reduction=reduction)
    
    def call(self, y_pred, y_true):
        distance_matrix = tf.sqrt( tf.reduce_sum( tf.square( tf.expand_dims(y_true,axis=-2) - y_pred), axis = -1 ) )
        distance_per_prediction = tf.reduce_min( distance_matrix, axis = -1 )
        
        return tf.reduce_mean(distance_per_prediction)

# 
# y_pred = np.array([[0.0, 1.0], [0.0, 4.0]])
# y_true = np.array([[0.0, 0.0], [0.0, 5.0], [2.0, 4.0]])
# metric = MeanDistanceToSet(reduction=tf.keras.losses.Reduction.NONE)
# print( metric(y_pred, y_true) )