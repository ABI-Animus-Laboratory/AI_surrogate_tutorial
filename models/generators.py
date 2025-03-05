'''
Created on 5/08/2020

@author: Gonzalo D. Maso Talou
'''

import tensorflow as tf
import numpy as np
from numpy import pi

class DualStiffnessBCGenerator(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.E1_MIN = 2.0
        self.E1_MAX = 5.0
        self.E2_MIN = 4.0
        self.E2_MAX = 40.0
        self.P_MIN = 0.0
        self.P_MAX = 0.75
        self.PCA_1_MIN = -2.0
        self.PCA_1_MAX = 2.0
        self.PCA_2_MIN = -2.0
        self.PCA_2_MAX = 2.0

        
    def generate(self,batch_size,condition_BD,values_BD):
        
        t = np.random.rand(batch_size,1)
        rad = values_BD[1]
        
        if condition_BD[0]:
            z = rad * np.cos(2*pi*t)
            y = rad * np.sin(2*pi*t)
            x = np.ones((batch_size,1)) * values_BD[0]
        else:
            if condition_BD[1]:
                z = rad * np.cos(2*pi*t)
                x = rad * np.sin(2*pi*t)
                y = np.ones((batch_size,1)) * values_BD[0]
            else:
                x = rad * np.cos(2*pi*t)
                y = rad * np.sin(2*pi*t)
                z = np.ones((batch_size,1)) * values_BD[0]
    
        
        E1 = np.random.rand(batch_size,1) * (self.E1_MAX - self.E1_MIN) + self.E1_MIN
        E2 = np.random.rand(batch_size,1) * (self.E2_MAX - self.E2_MIN) + self.E2_MIN
        P = np.random.rand(batch_size,1) * (self.P_MAX - self.P_MIN) + self.P_MIN
        PCA_1 = np.random.rand(batch_size,1) * (self.PCA_1_MAX - self.PCA_1_MIN) + self.PCA_1_MIN
        PCA_2 = np.random.rand(batch_size,1) * (self.PCA_2_MAX - self.PCA_2_MIN) + self.PCA_2_MIN
        inputs = tf.concat([x,y,z,E1,E2,P,PCA_1,PCA_2], axis = -1) 
    
        dx = np.zeros((batch_size,1))
        dy = np.zeros((batch_size,1))
        dz = np.zeros((batch_size,1))
        outputs = tf.concat([dx,dy,dz], axis = -1) 
        
        yield(inputs,outputs)
        

class SingleStiffnessBCGenerator(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.E1_MIN = 2.0
        self.E1_MAX = 5.0
        self.P_MIN = 0.0
        self.P_MAX = 0.75
        self.PCA_1_MIN = -2.0
        self.PCA_1_MAX = 2.0
        self.PCA_2_MIN = -2.0
        self.PCA_2_MAX = 2.0

        
    def generate(self,batch_size,condition_BD,values_BD):
        
        t = np.random.rand(batch_size,1)
        rad = values_BD[1]
        
        if condition_BD[0]:
            z = rad * np.cos(2*pi*t)
            y = rad * np.sin(2*pi*t)
            x = np.ones((batch_size,1)) * values_BD[0]
        else:
            if condition_BD[1]:
                z = rad * np.cos(2*pi*t)
                x = rad * np.sin(2*pi*t)
                y = np.ones((batch_size,1)) * values_BD[0]
            else:
                x = rad * np.cos(2*pi*t)
                y = rad * np.sin(2*pi*t)
                z = np.ones((batch_size,1)) * values_BD[0]
    
        
        E1 = np.random.rand(batch_size,1) * (self.E1_MAX - self.E1_MIN) + self.E1_MIN
        P = np.random.rand(batch_size,1) * (self.P_MAX - self.P_MIN) + self.P_MIN
        PCA_1 = np.random.rand(batch_size,1) * (self.PCA_1_MAX - self.PCA_1_MIN) + self.PCA_1_MIN
        PCA_2 = np.random.rand(batch_size,1) * (self.PCA_2_MAX - self.PCA_2_MIN) + self.PCA_2_MIN
        inputs = tf.concat([x,y,z,E1,P,PCA_1,PCA_2], axis = -1) 
    
        dx = np.zeros((batch_size,1))
        dy = np.zeros((batch_size,1))
        dz = np.zeros((batch_size,1))
        outputs = tf.concat([dx,dy,dz], axis = -1) 
        
        yield(inputs,outputs)