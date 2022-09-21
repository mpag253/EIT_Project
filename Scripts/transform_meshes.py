import os
import numpy as np
import argparse
import math
import pandas as pd
from sys import exit
   
#studies = ['Human_Aging']
#subjects = ['AGING006']
#protocols = ['EIsupine']

# cmd: python3 crop_torso_data.py


#def get_lung_node_coords(file, node_num):
#    file_n = open(file, 'r')
#    lines_n = file_n.readlines()
#    for ln, line in enumerate(lines_n):
#        if line.startswith("   z.  Value"):
#            splitline = line.split()
#            if splitline[-2] == "#Versions=":
#                nvers = int(line.split()[-1])
#            else:
#                nvers = 1
#        if line.startswith(" Node:"):
#            if int(line.split()[-1]) == node_num:
#                # identify the row and column of each value from the the exnode format
#                #zpos = [2*math.ceil(nvers*4/5.) + math.ceil((nvers*4-3)/5.), (nvers*4-3)%5-1]
#                xrow = int(0*math.ceil(nvers*4/5.) + math.ceil((nvers*4-3)/5.))
#                yrow = int(1*math.ceil(nvers*4/5.) + math.ceil((nvers*4-3)/5.))
#                zrow = int(2*math.ceil(nvers*4/5.) + math.ceil((nvers*4-3)/5.))
#                col = (nvers*4-3)%5-1
#                # retrieve the values
#                xval = float(lines_n[ln+xrow].split()[col])
#                yval = float(lines_n[ln+yrow].split()[col])
#                zval = float(lines_n[ln+zrow].split()[col])
#    file_n.close()
#    return [xval, yval, zval]

###
def generate_rotation_matrix(U,V):
    W=np.cross(U,V)
    A=np.array([U,W,np.cross(U,W)]).T
    B=np.array([V,W,np.cross(V,W)]).T
    return np.dot(B,np.linalg.inv(A))
    
###
def rotate_in_x(vector_1, ang):
    rotmat_x = [[ 1,           0,            0],
                [ 0, np.cos(ang), -np.sin(ang)],
                [ 0, np.sin(ang),  np.cos(ang)]]
    vector_2 = np.matmul(rotmat_x, vector_1)                
    return vector_2
    
    
###
def rotate_in_y(vector_1, ang):
    rotmat_y = [[  np.cos(ang), 0, np.sin(ang)],
                [            0, 1,           0],
                [ -np.sin(ang), 0, np.cos(ang)]]
    vector_2 = np.matmul(rotmat_y, vector_1)
    return vector_2
    

###
def rotate_in_z(vector_1, ang):
    rotmat_z = [[ np.cos(ang), -np.sin(ang), 0],
                [ np.sin(ang),  np.cos(ang), 0],
                [ 0,        0,               1]]
    vector_2 = np.matmul(rotmat_z, vector_1)
    return vector_2
    

###
def do_transform(unit_normal_vector, in_vector):

    # align to median plane
    angle_a = np.arctan(unit_normal_vector[2]/unit_normal_vector[0])
    vector_a = rotate_in_y(in_vector, angle_a)
    angle_b = -np.arcsin(unit_normal_vector[1]/1)
    vector_b = rotate_in_z(vector_a, angle_b)
    
    # correct z-direction
    unit_z = [0, 0, 1]
    unit_z_a = rotate_in_y(unit_z, angle_a)
    unit_z_b = rotate_in_z(unit_z_a, angle_b)
    correction_angle = np.arctan(unit_z_b[1]/unit_z_b[2])
    #unit_z_c = rotate_in_x(unit_z_b, correction_angle)
    vector_c = rotate_in_x(vector_b, correction_angle)
    
    #print("in_vector:", in_vector)
    #print("vector_b:", vector_b)
    #print("vector_a:", vector_a, "\n")
    #print("correction_angle:", correction_angle)
    #print("unit_z_b:", unit_z_b) 
    #print("unit_z_c:", unit_z_c)
    #print("vector_c:", vector_c)
    
    return vector_c
    
    
###
def get_voxel_coordinates(indices, image_info):
    """
    """
    #unpack image info
    [d1, d2, d3, p1, p2, p3] = image_info
    # generate the coordinates   
    x_coord = p1/2 + p1*indices[0]
    y_coord = -(p2/2 + p2*indices[1]) + d2*p2
    z_coord = p3/2 + p3*indices[2] - d3*p3
    return [x_coord, y_coord, z_coord]
    


#######################################

#pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'

#versions = ['mean', 
#            'mode01_pos2p5', 'mode01_neg2p5', 
#            'mode02_pos2p5', 'mode02_neg2p5',
#            'mode03_pos2p5','mode03_neg2p5',
#            'mode04_pos2p5','mode04_neg2p5',
#            'mode05_pos2p5','mode05_neg2p5']

pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A'
versions = ['sample_mean', 'predict_H5977']

bodies = ['Right', 'Left', 'Torso']

root = "/hpc/mpag253/Torso/SFEAL/"

# (optional for scale data)
scale_factor = 1.0

# Rotate the data using reorientation vector
normal_vector = [np.cos(0.0729), -np.sin(0.0729), 0]
unit_normal_vector = normal_vector/np.linalg.norm(normal_vector)   

## (test) 
#test_vector = do_transform(unit_normal_vector, unit_normal_vector)
#print("test vector:", test_vector)
#exit(0)

for version in versions:

    mesh_input_dir = os.path.join(root, 'output/export_to_cm/pca_'+pca_id, version)
    
    for body in bodies:
    
                
        # EXNODE: Read data, transform, and write new file
        #print('EXNODE:')
        for lung in ['Right', 'Left']:
            file_data = open(mesh_input_dir+"/fitted"+body+".exnode", 'r')
            lines_data = file_data.readlines()
            file_data.close()
            remove_gt_count = 0
            remove_lt_count = 0
            for ln, line in enumerate(lines_data):
            
                # current number of versions
                if line.startswith("   z.  Value"):
                    line_split = line.split()
                    if line_split[-2] == '#Versions=':
                        n_vers = int(line_split[-1])
                    else:
                        n_vers = 1
            
                if line.startswith(" Node:"):
                
                    # number of columns in file
                    n_cols = 5
                    # number of values for each direction
                    n_vals = n_vers*4
                    # number of rows for each direction
                    n_rows = int(np.ceil(n_vals/n_cols))
                    # number of remaining values in final row
                    n_rem = n_vals%n_cols
                    #print(n_vers, n_cols, n_vals, n_rows, n_rem)
                    
                    # assemble data for each direction
                    data = np.empty([3, n_vers, 4])
                    for d in range(3):
                        data_str = lines_data[ln+d*n_rows+1].split()
                        for j in range(1, n_rows):
                            data_str += lines_data[ln+d*n_rows+j+1].split()
                        data_float = [float(v) for v in data_str]
                        data[d] = np.reshape(data_float, [n_vers, 4])
                
                    # transform data
                    data_tf = np.empty(np.shape(data))
                    for ver in range(n_vers):
                        for deriv in range(4):
                            [xval, yval, zval] = data[:, ver, deriv]
                            # (optional scale data)
                            xval *= scale_factor
                            yval *= scale_factor
                            zval *= scale_factor
                            # rotational transform
                            [xp, yp, zp] = do_transform(unit_normal_vector, [xval, yval, zval])
                            data_tf[:, ver, deriv] = [xp, yp, zp]
                                                                  
                    # disassemble data for write
                    for d in range(3):
                        # data transformed 
                        dtff = data_tf[d].flatten()
                        # enter data for all lines except final
                        for j in range(n_rows-1):
                            format_str = " {:24e} {:24e} {:24e} {:24e} {:24e}\n"
                            k = j*5
                            lines_data[ln+d*n_rows+j+1] = format_str.format(dtff[k+0], dtff[k+1], dtff[k+2], dtff[k+3], dtff[k+4])
                    
                        # enter data for final line
                        k = (n_rows-1)*5
                        if n_rem == 0:
                            format_str = " {:24e} {:24e} {:24e} {:24e} {:24e}\n"
                            lines_data[ln+d*n_rows+n_rows] = format_str.format(dtff[k+0], dtff[k+1], dtff[k+2], dtff[k+3], dtff[k+4])
                        elif n_rem == 1:
                            format_str = " {:24e}\n"
                            lines_data[ln+d*n_rows+n_rows] = format_str.format(dtff[k+0])
                        elif n_rem == 2:
                            format_str = " {:24e} {:24e}\n"
                            lines_data[ln+d*n_rows+n_rows] = format_str.format(dtff[k+0], dtff[k+1])
                        elif n_rem == 3:
                            format_str = " {:24e} {:24e} {:24e}\n"
                            lines_data[ln+d*n_rows+n_rows] = format_str.format(dtff[k+0], dtff[k+1], dtff[k+2])
                        elif n_rem == 4:
                            format_str = " {:24e} {:24e} {:24e} {:24e}\n"
                            lines_data[ln+d*n_rows+n_rows] = format_str.format(dtff[k+0], dtff[k+1], dtff[k+2], dtff[k+3])
                        
                        # add whitespace to first line of each direction
                        lines_data[ln+d*n_vers+1] = "  "+lines_data[ln+d*n_vers+1]
             
            file_out = open(mesh_input_dir+"/fitted"+body+"_realigned.exnode", 'w')
            file_out.writelines(lines_data)
            file_out.close() 
        
       
       
       
#        # IPNODE: Read data, transform, and write new file
#        #print('IPNODE:')
#        for lung in ['Right', 'Left']:
#            file_data = open(mesh_input_dir+"/fitted"+body+".ipnode", 'r')
#            lines = file_data.readlines()
#            file_data.close()
#            remove_gt_count = 0
#            remove_lt_count = 0
#            for ln, line in enumerate(lines):
#                
#                # Extract the number of derivative - assumed equal for all directions
#                if line.startswith(" The number of derivatives for coordinate 1"):
#                    n_deriv = int(line.split()[-1])
#                
#                # Extract info for each node
#                if line.startswith(" Node number "):
#                
#                    print("---\n")
#                    
#                    # node number
#                    node_num = int(line.split()[-1])
#                    # number of versions
#                    n_vers = int(lines[ln+1].split()[-1])
#                    # number of node values
#                    n_vals = n_deriv + 1
#                
#                    # assemble data for each direction
#                    data = np.empty([n_vers, n_vals, 3])
#                    for v in range(n_vers):
#                        for d in range(n_vals):
#                            for c in range(3):
#                                if n_vers == 1:
#                                    data_ln = ln+2+(5*c)+d
#                                else:
#                                    data_ln = ln+3+(5*c*n_vers+c)+(5*v)+d
#                                split_line = lines[data_ln].split()
#                                print(split_line)
#                                data[v, d, c] = float(split_line[-1])
#                    #print(data)
#                                       
#                    # transform data
#                    data_tf = np.empty(np.shape(data))
#                    for v in range(n_vers):
#                        for d in range(n_vals):
#                            [xval, yval, zval] = data[v, d, :]
#                            # (optional scale data)
#                            xval *= scale_factor
#                            yval *= scale_factor
#                            zval *= scale_factor
#                            # rotational transform
#                            [xp, yp, zp] = do_transform(unit_normal_vector, [xval, yval, zval])
#                            data_tf[v, d, :] = [xp, yp, zp]
#                    #print(data_tf, "\n")
#                    
#                    # write out the transformed data
#                    for v in range(n_vers):
#                        for d in range(n_vals):
#                            for c in range(3):
#                                if n_vers == 1:
#                                    data_ln = ln+2+(5*c)+d
#                                else:
#                                    data_ln = ln+3+(5*c*n_vers+c)+(5*v)+d
#                                split_line = lines[data_ln].split()
#                                split_line[-1] = "{:>21f}\n".format(data_tf[v, d, c])
#                                lines[data_ln] = " "+" ".join(split_line)
#            
#            file_out = open(mesh_input_dir+"/fitted"+body+"_realigned.ipnode", 'w')
#            file_out.writelines(lines)
#            file_out.close() 
        
        
        
        
    
    
    
    
    
    
    
    
    
                