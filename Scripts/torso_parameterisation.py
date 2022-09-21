import openpyxl
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
import BSplineParameterisation.bspline_parameterisation as bsparam
#import scipy.interpolate as ip

# import shape_mismatch as sm
#from shapely.geometry import Polygon
#import os
import pickle

from Slice2D import assembly_plane_intersection_curve as assmcurve
from Slice2D import read_in_data as read_in_data

# # Import test geometries from torso STLs
# # path = os.path.join("C:\\","Users","mipag_000","Documents","EIT_Project","ShapeMismatch")
# path_data = ".\\ShapeMismatch\\Data"
# torso_sbj_file = 'ST4_E_boundary_sf0700.csv'
# torso_sbj_data = np.loadtxt(os.path.join(path_data, torso_sbj_file), delimiter=',')
# # # torso_pop_file = 'ST4_A_for_E_boundary_sf0700.csv'
# # torso_pop_file = 'ST4_A_boundary_sf0700.csv'
# # torso_pop_data = np.loadtxt(os.path.join(path_data, torso_pop_file), delimiter=',')
# # # print(torso_example_data)
#
# # # centre data on origin (assuming alignment of both, i.e., derived from same PCA model)
# # sbj_centre = -(np.min(torso_sbj_data, axis=0) + np.max(torso_sbj_data, axis=0))/2
# # # pop_centre = -(np.min(torso_pop_data, axis=0) + np.max(torso_pop_data, axis=0))/2
# # sbj_coords = torso_sbj_data + sbj_centre
# # # pop_coords = torso_pop_data + pop_centre
#
# # Convert to 3D coordinates for parameterisation function
# # sbj_coords = np.concatenate((sbj_coords, np.zeros((np.shape(sbj_coords)[0], 1))), axis=1)
# sbj_coords = np.concatenate((torso_sbj_data, np.zeros((np.shape(torso_sbj_data)[0], 1))), axis=1)
# # pop_coords = np.concatenate((pop_coords, np.zeros((np.shape(pop_coords)[0], 1))), axis=1)

########################################################################################################################
# MESH - POPULATION AVERAGE GEOMETRY ###################################################################################
# using example geometry for development

# # Define the number of knots (= number of EIT parameters) and create an equally spaced array of knots
# num_knots = 50
# knots = np.linspace(0*np.pi, 2*np.pi, num_knots+1)[0:-1]
#
# # # Convert subject geometry to polar
# # pop_polar = bsparam.get_polar_coordinates(pop_coords)
#
# # Get parameterised population torso shape
# # [Fs_mesh, phi_mesh] = bsparam.fit_mesh_bspline_to_data(pop_polar, knots)
# [Fs_mesh, _] = bsparam.get_parameterised_curve(pop_coords, knots, convert_to_rho=False, plot_bsplines=True)

torso_pop_file = 'bspdata_torso_mean.pkl'
with open(torso_pop_file, 'rb') as f:
    bsp_data_torso_mean = pickle.load(f)

knots = bsp_data_torso_mean['knots']
Fs_mesh = bsp_data_torso_mean['Fs']
pop_centre = bsp_data_torso_mean['centre']

########################################################################################################################
# SUBJECT - SPECIFIC GEOMETRY FOR PATIENT ##############################################################################

path = "C:\\Users\\mipag_000\\Documents\\EIT_Project\\"

# Define the filenames of the ".exnode" and ".exelem" files for each body
torso_sbj_file = 'TEST_HLA-H11303_predicted'
node_filenames = ["Geom\\"+torso_sbj_file+"\\fittedTorso.exnode"]
elem_filenames = ["Geom\\template\\templateTorso.exelem"]

# Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
plane = [0., 0., 1., 167.]  # [0., 0., 1., 120.]

# Define elements to ignore (e.g. lobe elements)
ignore_elems = []

# Define number of points to evaluate in each local element direction
nICpts = 100

nData = read_in_data.from_exnode(path, node_filenames)  # acting as a template
[eData, sData, aData] = read_in_data.from_exelem(path, elem_filenames)

# Get the intersection curves for all bodies and elements
int_curves_mean = assmcurve.get_intersection_curves(eData, aData, nData, plane, nICpts, ignore_elems,
                                                    plot_3d=True, plot_2d=True)

# Use shift parameter from population
sbj_coords = int_curves_mean[0][:, 2:5]
sbj_coords[:, :2] -= pop_centre

# convert subject geometry to polar
sbj_polar = bsparam.get_polar_coordinates(sbj_coords)

# evaluate population geometry radius, r(phi), for phi = subject geometry
# (could do alternative interpolation)
r_pop_phi_sbj = bsparam.evaluate_mesh_bspline(Fs_mesh[0], knots, sbj_polar[:, 0])

# convert the subject radius to rho-prime-space (rho space but for non-circular)
sbj_rho_coords = np.empty((np.shape(r_pop_phi_sbj)[0], 2))
sbj_rho_coords[:, 0] = sbj_polar[:, 0]
delta_r_sbj = sbj_polar[:, 1] - r_pop_phi_sbj
rho_factor = 0.5/max(np.absolute(delta_r_sbj))
sbj_rho_coords[:,1] = delta_r_sbj*rho_factor

# Generate B-spline for rho_sbj
[Fs_sbj, _] = bsparam.fit_mesh_bspline_to_data(sbj_rho_coords, knots)

# Plot
# plt.figure()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.fill(sbj_polar[:,0], sbj_polar[:,1], color=[0.9, 0.9, 0.9])
plt.plot(sbj_polar[:,0], sbj_polar[:,1], 'b')
plt.plot(sbj_polar[:,0], r_pop_phi_sbj, 'r')
plt.plot(sbj_polar[:,0], r_pop_phi_sbj + 0.5/rho_factor, 'r:')
plt.plot(sbj_polar[:,0], r_pop_phi_sbj - 0.5/rho_factor, 'r:')
plt.title('Subject (blue) and population (red) geometries')


plot_phis = np.linspace(0*np.pi, 2*np.pi, 1000)[0:-1]
rho_sbj_plot = bsparam.evaluate_mesh_bspline(Fs_sbj[0], knots, plot_phis)
r_pop_plot = bsparam.evaluate_mesh_bspline(Fs_mesh[0], knots, plot_phis)

plt.figure()
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.plot(sbj_rho_coords[:,0], sbj_rho_coords[:, 1], 'b')
plt.plot(plot_phis, rho_sbj_plot, 'c--')
plt.title('Subject geometry in rho\'-space')
plt.ylim([-.5, .5])
plt.grid()


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.fill(sbj_polar[:,0], sbj_polar[:,1], color=[0.9, 0.9, 0.9])
plt.plot(sbj_polar[:,0], sbj_polar[:,1], 'b')
plt.plot(plot_phis, r_pop_plot + rho_sbj_plot/rho_factor, 'c--')
plt.title('Subject original (blue) and parameterised (cyan) geometries')

plt.show()


# export [rho_factor, knots, Fs_sbj[0], knots]

# print(rho_factor)
# print(np.concatenate((np.reshape(knots, (np.shape(knots)[0], 1)), Fs_sbj[0]),axis=1))

# # EXPORT SUBJECT
bsparam.export_bspline('Torso', 'torso_sbj_test', Fs_sbj, knots, shift=pop_centre, rho_factor=rho_factor,
                       subject_name=torso_sbj_file, base_name=torso_pop_file)


