'''
Created on 11/08/2020

@author: Gonzalo D. Maso Talou
'''

import tensorflow as tf
import csv
import numpy as np

eps = 1E-12
eps2 = 1E-1

class CardiacInferenceReader(object):
    '''
    Reads a given set of files containing: (i) one reference configuration of the ventricle (zero-pressure 
    configuration) with the PCA weights that best project the image data; (ii) multiple time-points in the 
    spatial configuration.  
    '''


    def __init__(self, num_inputs, num_outputs, labels_ref, labels_data):
        '''
        Constructor
        Labels is the list of the CSV column headers. It should be sorted as num_inputs (x,y,z,parameters) first num_outputs last.
        '''
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.labels_ref = labels_ref
        self.labels_data = labels_data

    def read_reference(self, filename, path):
        """
        Returns an input vector as a tuple (x_learn, y_learn, z_learn, E1, E2, P, PCA1, PCA2) and a reference vector
        as a tuple (x_ref, y_ref, z_ref). the reference vector is the spatial configuration for zero-pressure 
        condition. 
        :param filename: filename containing the reference configuration data.
        :type filename: string
        :param path: folder containing the file with reference configuration data.
        :type path: string
        """
        file_inputs = []
        file_outputs = []
    
        print("Loading file {}".format(filename))
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:

            sample_inputs = []
            for idx in range(0,self.num_inputs):
                sample_inputs.append(float(sample[self.labels_ref[idx]]))
            file_inputs.append(sample_inputs)
            
            sample_reference = []
            for idx in range(0,self.num_outputs):
                sample_reference.append(float(sample[self.labels_ref[self.num_inputs+idx]]))
            file_outputs.append(sample_reference)
    
        inputs = np.array(file_inputs)    
        outputs = np.array(file_outputs)

        return inputs, outputs

    def read_data(self, filename, path, field_label):

        print("Loading file {}".format(filename))

        data = []
        value_data = []
        
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
        value = float('inf') 
        
        for sample in reader_data:
            if value != float(sample[field_label]):
                if value == float('inf'):
                    value_data = []
                else:
                    data.append(np.array(value_data))
                    value_data = []
            
                value = float(sample[field_label])
            
            sample_data = []
            for idx in range(0,self.num_inputs):
                sample_data.append(float(sample[self.labels_data[idx]]))
            value_data.append(sample_data)

        data.append(np.array(value_data))

        return data


class CardiacReader(object):
    '''
    Reads a given set of files containing data and labels for the surrogate. Data and labels are defined 
    over the same paired spatial points, i.e., for each point in those files it is also recorded its label.  
    '''


    def __init__(self, num_inputs, num_outputs, labels):
        '''
        Constructor
        Labels is the list of the CSV column headers. It should be sorted as num_inputs (x,y,z,parameters) first num_outputs last.
        '''
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.labels = labels
        
    def read_domain_values(self, filenames, path, condition, values):     

        file_inputs = []
        file_outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if not((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                    or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                    or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):

                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.labels[idx]]))
                    file_inputs.append(sample_inputs)
                    
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.labels[self.num_inputs+idx]]))
                    file_outputs.append(sample_outputs)
    
        inputs = np.array(file_inputs)    
        outputs = np.array(file_outputs)

        return inputs, outputs

    def read_neumann_boundary(self, filenames, path, condition, values):
        
        file_inputs = []
        file_outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                   or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                   or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):
                    
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.labels[idx]]))
                    file_inputs.append(sample_inputs)
                    
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.labels[self.num_inputs+idx]]))
                    file_outputs.append(sample_outputs)
    
        inputs = np.array(file_inputs)    
        outputs = np.array(file_outputs)

        return inputs, outputs
    
    def read_and_filtered_data(self, filenames, path, field_names, field_values, items_per_file=False):

        file_inputs = []
        file_outputs = []
        if items_per_file :
            inputs = []
            outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.labels[idx]]))
                    file_inputs.append(sample_inputs)
                    
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.labels[self.num_inputs+idx]]))
                    file_outputs.append(sample_outputs)
            
            if items_per_file :
                inputs.append(np.array(file_inputs))
                file_inputs = []
                outputs.append(np.array(file_outputs))
                file_outputs = []

        if not items_per_file :
            inputs = np.array(file_inputs)    
            outputs = np.array(file_outputs)

        return inputs, outputs

    def read_or_filtered_data(self, filenames, path, field_names, field_values, items_per_file=False):

        file_inputs = []
        file_outputs = []
        if items_per_file :
            inputs = []
            outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = False
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) < eps:
                        checks = True
                        break
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.labels[idx]]))
                    file_inputs.append(sample_inputs)
                    
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.labels[self.num_inputs+idx]]))
                    file_outputs.append(sample_outputs)
            
            if items_per_file :
                inputs.append(np.array(file_inputs))
                file_inputs = []
                outputs.append(np.array(file_outputs))
                file_outputs = []

        if not items_per_file :
            inputs = np.array(file_inputs)    
            outputs = np.array(file_outputs)

        return inputs, outputs


class UnpairCardiacReader(object):
    '''
    Reads a given set of files containing data and a different set of files containing labels for the surrogate. Data and labels are not necessarily defined over the spatial points. 
    '''


    def __init__(self, num_inputs, num_outputs, input_labels, output_labels):
        '''
        Constructor
        Labels is the list of the CSV column headers. It should be sorted as num_inputs (x,y,z,parameters) first num_outputs last.
        '''
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_labels = input_labels
        self.output_labels = output_labels
        
    def read_domain_values(self, filenames, path, condition, values):     

        file_inputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if not((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                    or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                    or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):

                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs.append(sample_inputs)
    
        inputs = np.array(file_inputs)    

        return inputs

    def read_domain_labels(self, filenames, path):     

        file_outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                sample_outputs = []
                for idx in range(0,self.num_outputs):
                    sample_outputs.append(float(sample[self.output_labels[idx]]))
                file_outputs.append(sample_outputs)
    
        outputs = np.array(file_outputs)

        return outputs

    def read_neumann_boundary(self, filenames, path, condition, values):
        
        file_inputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                   or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                   or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):
                    
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs.append(sample_inputs)
    
        inputs = np.array(file_inputs)    

        return inputs
    
    def read_and_filtered_data(self, filenames, path, field_names, field_values):

        file_inputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs.append(sample_inputs)
                    
        inputs = np.array(file_inputs)    

        return inputs

    def read_filtered_labels(self, filenames, path, field_names, field_values):

        file_outputs = []
    
        idx_file=1
        for filename in filenames: 
            print("Loading file ({}/{}): {}".format(idx_file,len(filenames),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.output_labels[idx]]))
                    file_outputs.append(sample_outputs)
                    
        inputs = np.array(file_outputs)    

        return inputs


class UnpairDualSurfaceReader(object):
    '''
    Reads a given set of files containing data and a different set of files containing labels for the surrogate. Data and labels are not necessarily defined over the spatial points. 
    '''


    def __init__(self, num_inputs, num_outputs, input_labels, output_labels):
        '''
        Constructor
        Labels is the list of the CSV column headers. It should be sorted as num_inputs (x,y,z,parameters) first num_outputs last.
        '''
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_labels = input_labels
        self.output_labels = output_labels
        
    def read_domain_values(self, filenames_surface_1, filenames_surface_2, path, condition, values):     

        file_inputs_1 = []
        file_inputs_2 = []
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if not((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                    or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                    or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):

                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_1.append(sample_inputs)
    
        inputs_1 = np.array(file_inputs_1)    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if not((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                    or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                    or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):

                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_2.append(sample_inputs)
    
        inputs_2 = np.array(file_inputs_2)    

        return inputs_1, inputs_2

    def read_domain_labels(self, filenames_surface_1, filenames_surface_2, path):     

        file_outputs_1 = []
        file_outputs_2 = []
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                sample_outputs = []
                for idx in range(0,self.num_outputs):
                    sample_outputs.append(float(sample[self.output_labels[idx]]))
                file_outputs_1.append(sample_outputs)
    
        outputs_1 = np.array(file_outputs_1)

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                sample_outputs = []
                for idx in range(0,self.num_outputs):
                    sample_outputs.append(float(sample[self.output_labels[idx]]))
                file_outputs_2.append(sample_outputs)
    
        outputs_2 = np.array(file_outputs_2)

        return outputs_1, outputs_2

    def read_neumann_boundary(self, filenames_surface_1, filenames_surface_2, path, condition, values):
        
        file_inputs_1 = []
        file_inputs_2 = []
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                   or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                   or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):
                    
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_1.append(sample_inputs)
    
        inputs_1 = np.array(file_inputs_1)    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                if((condition[0] and abs(float(sample[self.labels[0]])-values[0]) < eps) 
                   or (condition[1] and abs(float(sample[self.labels[1]])-values[1]) < eps) 
                   or (condition[2] and abs(float(sample[self.labels[2]])-values[2]) < eps)):
                    
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_2.append(sample_inputs)
    
        inputs_2 = np.array(file_inputs_2)    

        return inputs_1, inputs_2
    
    def read_and_filtered_data(self, filenames_surface_1, filenames_surface_2, path, field_names, field_values):

        file_inputs_1 = []
        file_inputs_2 = []
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_1.append(sample_inputs)
                    
        inputs_1 = np.array(file_inputs_1)    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    file_inputs_2.append(sample_inputs)
                    
        inputs_2 = np.array(file_inputs_2)    

        return inputs_1, inputs_2

    def read_and_filtered_labels(self, filenames_surface_1, filenames_surface_2, path, field_names, field_values):

        file_outputs_1 = []
        file_outputs_2 = []
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.output_labels[idx]]))
                    file_outputs_1.append(sample_outputs)
                    
        outputs_1 = np.array(file_outputs_1)    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = True
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) > eps:
                        checks = False
                        break
                    
                if checks:
                        
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.output_labels[idx]]))
                    file_outputs_2.append(sample_outputs)
                    
        outputs_2 = np.array(file_outputs_2)    

        return outputs_1, outputs_2

    def read_or_filtered_data(self, filenames_surface_1, filenames_surface_2, path, field_names, field_values):

        file_inputs_1 = []
        file_inputs_2 = []
        for idx_condition in range(0,len(field_values)):
            file_inputs_1.append([])
            file_inputs_2.append([])
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = False
                idx_checked = []
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) < eps:
                        checks = True
                        idx_checked.append(idx_condition)
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    for idx in idx_checked:
                        file_inputs_1[idx].append(sample_inputs)
        
        inputs_1 = []
        for idx_condition in range(0,len(field_values)):
            inputs_1.append(np.array(file_inputs_1[idx_condition]))    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = False
                idx_checked = []
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) < eps:
                        checks = True
                        idx_checked.append(idx_condition)
                    
                if checks:
                        
                    sample_inputs = []
                    for idx in range(0,self.num_inputs):
                        sample_inputs.append(float(sample[self.input_labels[idx]]))
                    for idx in idx_checked:
                        file_inputs_2[idx].append(sample_inputs)
                    
        inputs_2 = []
        for idx_condition in range(0,len(field_values)):
            inputs_2.append(np.array(file_inputs_1[idx_condition]))    

        return inputs_1, inputs_2

    def read_or_filtered_labels(self, filenames_surface_1, filenames_surface_2, path, field_names, field_values):

        file_outputs_1 = []
        file_outputs_2 = []
        for idx_condition in range(0,len(field_values)):
            file_outputs_1.append([])
            file_outputs_2.append([])
    
        idx_file=1
        for filename in filenames_surface_1: 
            print("Loading file for surface 1 ({}/{}): {}".format(idx_file,len(filenames_surface_1),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = False
                idx_checked = []
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) < eps:
                        checks = True
                        idx_checked.append(idx_condition)
                    
                if checks:
                        
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.output_labels[idx]]))
                    for idx in idx_checked:
                        file_outputs_1[idx].append(sample_outputs)
                    
        outputs_1 = []
        for idx_condition in range(0,len(field_values)):
            outputs_1.append(np.array(file_outputs_1[idx_condition]))    

        idx_file=1
        for filename in filenames_surface_2: 
            print("Loading file for surface 2 ({}/{}): {}".format(idx_file,len(filenames_surface_2),filename))
            idx_file = idx_file + 1
            reader_data = csv.DictReader(open(path+filename), delimiter=',')
        
            for sample in reader_data:
                checks = False
                idx_checked = []
                for idx_condition in range(0,len(field_values)):
                    if abs(float(sample[field_names[idx_condition]])-field_values[idx_condition]) < eps:
                        checks = True
                        idx_checked.append(idx_condition)
                    
                if checks:
                        
                    sample_outputs = []
                    for idx in range(0,self.num_outputs):
                        sample_outputs.append(float(sample[self.output_labels[idx]]))
                    for idx in idx_checked:
                        file_outputs_2[idx].append(sample_outputs)
                    
        outputs_2 = []
        for idx_condition in range(0,len(field_values)):
            outputs_2.append(np.array(file_outputs_2[idx_condition]))    

        return outputs_1, outputs_2
