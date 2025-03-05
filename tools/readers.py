'''
Created on 19/07/2019

@author: Gonzalo D. Maso Talou
'''
import csv
import numpy as np
from cmath import sqrt

eps = 1E-12
eps2 = 1E-1

def read_domain_data2(filenames, path, condition, values, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E1_train = []
    E2_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
        
    dx_val = []
    dy_val = []
    dz_val = []

    idx=1
    for filename in filenames: 
        print("Loading file ({}/{}): {}".format(idx,len(filenames),filename))
        idx = idx + 1
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if not((condition[0] and abs(float(sample['x'])-values[0]) < eps) or (condition[1] 
                and abs(float(sample['y'])-values[1]) < eps) or (condition[2] 
                and abs(float(sample['z'])-values[2]) < eps)):
                x_val.append(float(sample['x']))
                y_val.append(float(sample['y']))
                z_val.append(float(sample['z']))
                E1_train.append(float(sample['E1']))
                E2_train.append(float(sample['E2']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x\'']))
                    ys_train.append(float(sample['y\'']))
                    zs_train.append(float(sample['z\'']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E1_train)[:,None], np.array(E2_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E1_train)[:,None], np.array(E2_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]

def read_BN_data2(filenames, path, condition, values, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E1_train = []
    E2_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
    dx_val = []
    dy_val = []
    dz_val = []

    idx=1
    for filename in filenames: 
        print("Loading file ({}/{}): {}".format(idx,len(filenames),filename))
        idx = idx + 1
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if((condition[0] and abs(float(sample['x'])-values[0]) < eps) 
               or (condition[1] and abs(float(sample['y'])-values[1]) < eps) 
               or (condition[2] and abs(float(sample['z'])-values[2]) < eps)):
                x_val.append(float(sample['x']))
                y_val.append(float(sample['y']))
                z_val.append(float(sample['z']))
                E1_train.append(float(sample['E1']))
                E2_train.append(float(sample['E2']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x\'']))
                    ys_train.append(float(sample['y\'']))
                    zs_train.append(float(sample['z\'']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E1_train)[:,None], np.array(E2_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E1_train)[:,None], np.array(E2_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]


def read_domain_data(filenames, path, condition, values, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E_train = []
    E_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
        
    dx_val = []
    dy_val = []
    dz_val = []

    idx=1
    for filename in filenames: 
        print("Loading file ({}/{}): {}".format(idx,len(filenames),filename))
        idx = idx + 1
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if not((condition[0] and abs(float(sample['x_l'])-values[0]) < eps) or (condition[1] and abs(float(sample['y_l'])-values[1]) < eps) or (condition[2] and abs(float(sample['z_l'])-values[2]) < eps)):
                x_val.append(float(sample['x_l']))
                y_val.append(float(sample['y_l']))
                z_val.append(float(sample['z_l']))
                E_train.append(float(sample['E']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x_s']))
                    ys_train.append(float(sample['y_s']))
                    zs_train.append(float(sample['z_s']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]

def read_PCA_domain_data(filenames, ref_file, path, condition_BD, values_BD, condition_BN, values_BN, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
        
    dx_val = []
    dy_val = []
    dz_val = []
    
    xl_to_xPCA = {}

    reader_data_ref = csv.DictReader(open(path+ref_file), delimiter=',')
    for sample in reader_data_ref:
        x_l = sample['x_l']
        y_l = sample['y_l']
        z_l = sample['z_l']

        x_s = sample['x_s']
        y_s = sample['y_s']
        z_s = sample['z_s']
        xl_to_xPCA[(x_l,y_l,z_l)] = (x_s,y_s,z_s)


    for filename in filenames:
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if not((condition_BN[0] and abs(float(sample['x_l'])-values_BN[0]) < eps) 
                   or (condition_BN[1] and abs(float(sample['y_l'])-values_BN[1]) < eps) 
                   or (condition_BN[2] and abs(float(sample['z_l'])-values_BN[2]) < eps)):
                if not((condition_BD[0] and abs(float(sample['x_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['y_l'])**2 + float(sample['z_l'])**2) - values_BD[1]) < eps2) ) 
                       or (condition_BD[1] and abs(float(sample['y_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['x_l'])**2 + float(sample['z_l'])**2) - values_BD[1]) < eps2) ) 
                       or (condition_BD[2] and abs(float(sample['z_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['y_l'])**2 + float(sample['x_l'])**2) - values_BD[1]) < eps2) )):
                    xPCA = xl_to_xPCA[(sample['x_l'],sample['y_l'],sample['z_l'])]
                    x_val.append(float(xPCA[0]))
                    y_val.append(float(xPCA[1]))
                    z_val.append(float(xPCA[2]))
                    E_train.append(float(sample['E']))
                    P_train.append(float(sample['P']))
                    PCA1_train.append(float(sample['PCA_1']))
                    PCA2_train.append(float(sample['PCA_2']))
                    if spatial_coord:
                        xs_train.append(float(sample['x_s']))
                        ys_train.append(float(sample['y_s']))
                        zs_train.append(float(sample['z_s']))
                    dx_val.append(float(sample['dx']))
                    dy_val.append(float(sample['dy']))
                    dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]


def read_BN_data(filenames, path, condition, values, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
    dx_val = []
    dy_val = []
    dz_val = []

    idx=1
    for filename in filenames: 
        print("Loading file ({}/{}): {}".format(idx,len(filenames),filename))
        idx = idx + 1
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if((condition[0] and abs(float(sample['x_l'])-values[0]) < eps) 
               or (condition[1] and abs(float(sample['y_l'])-values[1]) < eps) 
               or (condition[2] and abs(float(sample['z_l'])-values[2]) < eps)):
                x_val.append(float(sample['x_l']))
                y_val.append(float(sample['y_l']))
                z_val.append(float(sample['z_l']))
                E_train.append(float(sample['E']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x_s']))
                    ys_train.append(float(sample['y_s']))
                    zs_train.append(float(sample['z_s']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]

def read_BN_PCA_data(filenames, ref_file, path, condition, values, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
    dx_val = []
    dy_val = []
    dz_val = []

    xl_to_xPCA = {}

    reader_data_ref = csv.DictReader(open(path+ref_file), delimiter=',')
    for sample in reader_data_ref:
        x_l = sample['x_l']
        y_l = sample['y_l']
        z_l = sample['z_l']

        x_s = sample['x_s']
        y_s = sample['y_s']
        z_s = sample['z_s']
        xl_to_xPCA[(x_l,y_l,z_l)] = (x_s,y_s,z_s)

    for filename in filenames: 
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if((condition[0] and abs(float(sample['x_l'])-values[0]) < eps) 
               or (condition[1] and abs(float(sample['y_l'])-values[1]) < eps) 
               or (condition[2] and abs(float(sample['z_l'])-values[2]) < eps)):
                xPCA = xl_to_xPCA[(sample['x_l'],sample['y_l'],sample['z_l'])]
                x_val.append(float(xPCA[0]))
                y_val.append(float(xPCA[1]))
                z_val.append(float(xPCA[2]))
                E_train.append(float(sample['E']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x_s']))
                    ys_train.append(float(sample['y_s']))
                    zs_train.append(float(sample['z_s']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]

def read_BD_PCA_data(filenames, ref_file, path, condition_BD, values_BD, spatial_coord=False):

    x_val = []
    y_val = []
    z_val = []
    E_train = []
    P_train = []
    PCA1_train = []
    PCA2_train = []
    if spatial_coord:
        xs_train = []
        ys_train = []
        zs_train = []
    dx_val = []
    dy_val = []
    dz_val = []

    xl_to_xPCA = {}

    reader_data_ref = csv.DictReader(open(path+ref_file), delimiter=',')
    for sample in reader_data_ref:
        x_l = sample['x_l']
        y_l = sample['y_l']
        z_l = sample['z_l']

        x_s = sample['x_s']
        y_s = sample['y_s']
        z_s = sample['z_s']
        xl_to_xPCA[(x_l,y_l,z_l)] = (x_s,y_s,z_s)

    for filename in filenames: 
        reader_data = csv.DictReader(open(path+filename), delimiter=',')
    
        for sample in reader_data:
            if ((condition_BD[0] and abs(float(sample['x_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['y_l'])**2 + float(sample['z_l'])**2) - values_BD[1]) < eps2) ) 
                or (condition_BD[1] and abs(float(sample['y_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['x_l'])**2 + float(sample['z_l'])**2) - values_BD[1]) < eps2) ) 
                or (condition_BD[2] and abs(float(sample['z_l'])-values_BD[0]) < eps and ( abs( sqrt( float(sample['y_l'])**2 + float(sample['x_l'])**2) - values_BD[1]) < eps2) ) ):
                xPCA = xl_to_xPCA[(sample['x_l'],sample['y_l'],sample['z_l'])]
                x_val.append(float(xPCA[0]))
                y_val.append(float(xPCA[1]))
                z_val.append(float(xPCA[2]))
                E_train.append(float(sample['E']))
                P_train.append(float(sample['P']))
                PCA1_train.append(float(sample['PCA_1']))
                PCA2_train.append(float(sample['PCA_2']))
                if spatial_coord:
                    xs_train.append(float(sample['x_s']))
                    ys_train.append(float(sample['y_s']))
                    zs_train.append(float(sample['z_s']))
                dx_val.append(float(sample['dx']))
                dy_val.append(float(sample['dy']))
                dz_val.append(float(sample['dz']))

    if spatial_coord:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(xs_train)[:,None], np.array(ys_train)[:,None], np.array(zs_train)[:,None], np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
    else:
        return np.array(x_val)[:,None], np.array(y_val)[:,None], np.array(z_val)[:,None], np.array(E_train)[:,None], np.array(P_train)[:,None], np.array(PCA1_train)[:,None], np.array(PCA2_train)[:,None], \
               np.array(dx_val)[:,None], np.array(dy_val)[:,None], np.array(dz_val)[:,None]
