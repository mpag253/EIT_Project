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
path_data = ".\\Data"
model_fwd = 'ST4_E_boundary_sf0700.csv'
model_inv = 'ST4_G_for_E_boundary_sf0700.csv'
geom_fwd = np.loadtxt(os.path.join(path_data, model_fwd), delimiter=',')
geom_inv = np.loadtxt(os.path.join(path_data, model_inv), delimiter=',')

# Define names for the plot
mdl_string_fwd = '_'.join(model_fwd.split('_')[:-2])
mdl_string_inv = '_'.join(model_inv.split('_')[:-2])
mdl_names = ['E','GforE']

sm.get_shape_mismatch(geom_fwd, geom_inv, geom_names=mdl_names, show_plot=True, save_plot=True)