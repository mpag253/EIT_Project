

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path as pth

pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A'
# version = 'truth_H5977'
# trimesh_file_path = 'Geom\\pca_'+pca_id+'\\'+version+'\\trimesh\\trimesh_0060_0015\\'
version = 'sample_mean'
trimesh_file_path = 'Geom\\pca_'+pca_id+'\\'+version+'\\trimesh\\trimesh_0080_0020\\'
trimesh_nodes_file = trimesh_file_path + 'trimesh_nodes.csv'
trimesh_tris_file = trimesh_file_path + 'trimesh_tris.csv'
trimesh_nodes = np.genfromtxt(trimesh_nodes_file, delimiter=',')
trimesh_tris = np.genfromtxt(trimesh_tris_file, delimiter=',').astype(int)-1

# # load lung seeds
# seeds_file_path = 'Geom\\pca_'+pca_id+'\\'+version+'\\mesh_seeds\\'
# seeds_file_rght = seeds_file_path + 'mesh_seeds_lung_right_0060.csv'
# seeds_file_left = seeds_file_path + 'mesh_seeds_lung_left_0060.csv'
# lung_seeds_rght = np.loadtxt(seeds_file_rght, delimiter=",")
# lung_seeds_left = np.loadtxt(seeds_file_left, delimiter=",")

# # get the lung nodes
# lung_path_rght = pth.Path(lung_seeds_rght)
# lung_nodes_rght = lung_path_rght.contains_points(trimesh_nodes, radius=1.)
# lung_path_left = pth.Path(lung_seeds_left)
# lung_nodes_left = lung_path_left.contains_points(trimesh_nodes, radius=1.)
# lung_nodes = lung_nodes_rght + lung_nodes_left

# constant
# lung_nodes = 0.5*np.ones(np.shape(lung_nodes))
# lung_nodes = 1.0*np.ones(np.shape(lung_nodes))
# lung_nodes = 0.5*np.ones(np.shape(trimesh_nodes[:,0]))
lung_nodes = 1.0*np.ones(np.shape(trimesh_nodes[:,0]))

# 2D - MEAN
fig, ax = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
ax.plot_trisurf(trimesh_nodes[:, 0], trimesh_nodes[:, 1], trimesh_tris, lung_nodes,
                cmap='viridis', linewidth=0, antialiased=False,
                vmin=0, vmax=1)
ax.view_init(azim=90, elev=90)
ax.set_box_aspect((np.ptp(trimesh_nodes[:, 0]),
                   np.ptp(trimesh_nodes[:, 1]),
                   1))  # aspect ratio is 1:1:1 in data space
ax.set_axis_off()
plt.show()