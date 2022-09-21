import shape_mismatch as sm
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os

# Simple test geometries
# geom_fwd = np.array([[1, 1], [2, 2], [4, 2], [3, 1]])
# geom_inv = np.array([[1.5, 2], [3, 5], [5, 4], [3.5, 1]])

# Import test geometries from torso STLs
# path = os.path.join("C:\\","Users","mipag_000","Documents","EIT_Project","ShapeMismatch")
pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A'
subject_ids = {'A':'H5977', 'B':'AGING043', 'C':'H7395', 'D':'AGING014', 'E':'AGING053'}
subject_id = subject_ids[pca_id.split('-')[-1]]
path_data = "..\\Geom\\pca_"+pca_id+"\\"
#
torso_file_fwd = 'truth_'+subject_id+'\\mesh_seeds\\mesh_seeds_torso_0015.csv'
lungl_file_fwd = 'truth_'+subject_id+'\\mesh_seeds\\mesh_seeds_lung_left_0015.csv'
lungr_file_fwd = 'truth_'+subject_id+'\\mesh_seeds\\mesh_seeds_lung_right_0015.csv'
torso_file_inv = 'sample_mean\\mesh_seeds\\mesh_seeds_torso_0020.csv'
lungl_file_inv = 'sample_mean\\mesh_seeds\\mesh_seeds_lung_left_0020.csv'
lungr_file_inv = 'sample_mean\\mesh_seeds\\mesh_seeds_lung_right_0020.csv'
# torso_file_inv = 'predicted_'+subject_id+'\\mesh_seeds\\mesh_seeds_torso_0020.csv'
# lungl_file_inv = 'predicted_'+subject_id+'\\mesh_seeds\\mesh_seeds_lung_left_0020.csv'
# lungr_file_inv = 'predicted_'+subject_id+'\\mesh_seeds\\mesh_seeds_lung_right_0020.csv'

torso_fwd = np.loadtxt(os.path.join(path_data, torso_file_fwd), delimiter=',')
lungl_fwd = np.flipud(np.loadtxt(os.path.join(path_data, lungl_file_fwd), delimiter=','))
lungr_fwd = np.flipud(np.loadtxt(os.path.join(path_data, lungr_file_fwd), delimiter=','))
torso_inv = np.loadtxt(os.path.join(path_data, torso_file_inv), delimiter=',')
lungl_inv = np.flipud(np.loadtxt(os.path.join(path_data, lungl_file_inv), delimiter=','))
lungr_inv = np.flipud(np.loadtxt(os.path.join(path_data, lungr_file_inv), delimiter=','))

# print(geom_fwd)
# print(geom_inv)

# plt.figure()
# plt.plot(lungl_fwd[:,0],lungl_fwd[:,1])
# plt.plot(lungl_fwd[0,0],lungl_fwd[0,1],'o')
# plt.plot(lungr_fwd[:,0],lungr_fwd[:,1])
# plt.plot(lungr_fwd[0,0],lungr_fwd[0,1],'o')
# plt.figure()
# plt.plot(lungl_inv[:,0],lungl_inv[:,1])
# plt.plot(lungl_inv[0,0],lungl_inv[0,1],'o')
# plt.plot(lungr_inv[:,0],lungr_inv[:,1])
# plt.plot(lungr_inv[0,0],lungr_inv[0,1],'o')
# plt.show()


# RMSE

# torso
theta_fwd = np.arctan2(torso_fwd[:,1],torso_fwd[:,0])
theta_inv = np.arctan2(torso_inv[:,1],torso_inv[:,0])
tx_inv_resamp = np.interp(theta_fwd, theta_inv, torso_inv[:,0], period=2*np.pi)
ty_inv_resamp = np.interp(theta_fwd, theta_inv, torso_inv[:,1], period=2*np.pi)
torso_inv_resamp = np.vstack((tx_inv_resamp,ty_inv_resamp)).T

def get_cumperim(seeds):
    dx = np.concatenate((seeds[1:,0], [seeds[0,0]]))-seeds[:,0]
    dy = np.concatenate((seeds[1:,1], [seeds[0,1]]))-seeds[:,1]
    perims = np.sqrt(np.square(dx) + np.square(dy))
    cumperim = np.cumsum(perims)
    cumperim /= cumperim[-1]
    return cumperim

# lung left
param_fwd = get_cumperim(lungl_fwd)
param_inv = get_cumperim(lungl_inv)
llx_inv_resamp = np.interp(param_fwd, param_inv, lungl_inv[:,0], period=1)
lly_inv_resamp = np.interp(param_fwd, param_inv, lungl_inv[:,1], period=1)
lungl_inv_resamp = np.vstack((llx_inv_resamp,lly_inv_resamp)).T

# lung right
param_fwd = get_cumperim(lungr_fwd)
param_inv = get_cumperim(lungr_inv)
lrx_inv_resamp = np.interp(param_fwd, param_inv, lungr_inv[:,0], period=1)
lry_inv_resamp = np.interp(param_fwd, param_inv, lungr_inv[:,1], period=1)
lungr_inv_resamp = np.vstack((lrx_inv_resamp,lry_inv_resamp)).T

all_fwd = np.vstack((torso_fwd,lungl_fwd,lungr_fwd))
all_inv_resamp = np.vstack((torso_inv_resamp,lungl_inv_resamp,lungr_inv_resamp))

diff = all_inv_resamp-all_fwd
dists = np.linalg.norm(diff,axis=1)
rmse = np.sqrt(np.mean(np.square(dists)))
print(rmse)

# SHAPE MISMATCH
# Define names for the plot
mdl_string_fwd = '_'.join(torso_file_fwd.split('_')[:-2])
mdl_string_inv = '_'.join(torso_file_inv.split('_')[:-2])
mdl_names = ['E','GforE']
# Run SM
sm.get_shape_mismatch(torso_fwd, torso_inv, geom_names=mdl_names, show_plot=True, save_plot=True)
