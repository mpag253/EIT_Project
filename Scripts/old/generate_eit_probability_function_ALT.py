# import openpyxl
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.path as pth
# from matplotlib import cm
# import Slice2D.element_plane_intersection_curve as elemcurve
import Slice2D.assembly_plane_intersection_curve as assmcurve
import Slice2D.read_in_data as read_in_data
import BSplineParameterisation.bspline_parameterisation as bsparam
# import scipy.interpolate as ip
# import time
from sys import exit
import copy
import pickle
# from scipy import stats
from scipy import sparse
import scipy.io
from matplotlib import path as pth
import parameterise_torso as parat

# INPUTS ###############################################################################################################

path = "/\\"

pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
mesh_id = 'trimesh_nodes_0010'

# Conditional covariance of MLR mode shapes
Gamma_wgivenp = np.load("Geom\\pca_"+pca_id+"\\pca_condcov_wgivenp_"+pca_id+".npy")

# Parameterised shapes
# Gamma_bgivenp = np.load('Geom\\pca_'+pca_id+'\\bspline_condcov.npy')
# Gamma_bgivenp += 1e-6*np.eye(len(Gamma_bgivenp))
bspline_data = np.load('Geom\\pca_'+pca_id+'\\bspline_data.npy')
# [Is_body, Js_body, Fs_mean, Ks_mean] = bspline_data.T
# print(Is_body, Js_body, Fs_mean, Ks_mean)
# temp
[Is_body, Fs_mean, Ks_mean] = bspline_data.T
Js_body = np.ones([len(Is_body)])
Js_body[22:44] = 2
Js_body[66:88] = 2
# print(Is_body, Js_body, Fs_mean, Ks_mean)

mode_splines = np.load('Geom\\pca_'+pca_id+'\\mode_splines.npy')
n_modes = len(mode_splines.T)

[lung_index_r, lung_index_l] = [0, 1]
lung_indices = [lung_index_r, lung_index_l]
n_evals = 1000
eval_phis = [np.linspace(0, 1, n_evals),
             np.linspace(0, 1, n_evals),
             np.linspace(0, 2*np.pi, n_evals)]

# Bayesian
n_samples = 10

# Mesh
base = 'TEST_population_sample_mean'
base_mesh_nodes = np.genfromtxt('Geom\\' + base + '\\'+mesh_id+'.csv', delimiter=',')
n_nodes = len(base_mesh_nodes)
# base_mesh_tris = np.genfromtxt('Geom\\'+base+'\\trimesh_tris_0010.csv', delimiter=',')
with open('Geom\\' + base + '\\bspdata_torso.pkl', 'rb') as f:
    bsp_data = pickle.load(f)
Ks_base_mesh = bsp_data['knots']
Fs_base_mesh = bsp_data['Fs'][0]
# base_mesh_centre = bsp_data['centre']
# print(Ks_base_mesh)
# print(Fs_base_mesh)

# Setup
bodies = np.unique(Is_body).astype(int)
body_dims = [np.unique(Js_body[Is_body==i]).astype(int) for i in bodies]
n_body_dims = np.sum([len(i) for i in body_dims])
show_plots = False


def pickle_dump(filename, data):
    # Save data to file (pickle)
    pkl_file = open(filename + '.pkl', "wb")
    pickle.dump(data, pkl_file)
    pkl_file.close()
    print("\nSaved data to pickle: ", filename + '.pkl')


def test_complete_intcurves(int_curves):
    """Test if the intersection curve is complete"""

    accepted = True
    for i in range(len(int_curves)):
        int_curve = int_curves[i]
        if not np.all(np.isclose(int_curve[0, 2:], int_curve[-1, 2:], atol=1e-01)):
            accepted = False

    return accepted


def distort_mesh(Fs_sbj, Ks_sbj, sbj_mesh_centre=None):
    """distort the mesh nodes based on fitted and imported bsplines"""

    # # fit bspline to subject torso
    # body_coords = int_curves[2][:, 2:]
    # if sbj_mesh_centre is not None:
    #     body_coords[:, :2] -= sbj_mesh_centre
    # # print(body_coords)
    # plt.close('all')
    # knots_sbj = np.linspace(0, 2*np.pi, 100)[1:]
    # [Fs_sbj, _] = bsparam.get_parameterised_curve(body_coords, knots_sbj, convert_polar=True, convert_rho=False,
    #                                               discont_elems=None, plot_bsplines=False, plot_bases=False)

    # angular position of each node
    node_theta = np.arctan2(base_mesh_nodes[:, 1], base_mesh_nodes[:, 0]) + np.pi
    # radius of each node
    node_base_r = np.sqrt(np.square(base_mesh_nodes[:, 1]) + np.square(base_mesh_nodes[:, 0]))
    # base torso radius at each node angular position
    node_base_ro = bsparam.evaluate_mesh_bspline(Fs_base_mesh, Ks_base_mesh, node_theta)
    # sampled torso radius at each node angular position
    # print(Fs_sbj)
    # print(Ks_sbj)
    node_sbj_ro = bsparam.evaluate_mesh_bspline(Fs_sbj, Ks_sbj, node_theta)

    # new node radius
    node_sbj_r = np.multiply(node_base_r, np.divide(node_sbj_ro, node_base_ro))
    # convert back to cartesian
    # node_base_x = -np.multiply(node_base_r, np.cos(node_theta))
    # node_base_y = -np.multiply(node_base_r, np.sin(node_theta))
    # node_base_xo = -np.multiply(node_base_ro, np.cos(node_theta))
    # node_base_yo = -np.multiply(node_base_ro, np.sin(node_theta))
    # node_sbj_xo = -np.multiply(node_sbj_ro, np.cos(node_theta))
    # node_sbj_yo = -np.multiply(node_sbj_ro, np.sin(node_theta))
    node_sbj_x = -np.multiply(node_sbj_r, np.cos(node_theta))
    node_sbj_y = -np.multiply(node_sbj_r, np.sin(node_theta))
    #
    # # plot check
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(body_coords[:, 0], body_coords[:, 1], color="blue")  # blue
    # ax1.scatter(node_base_x, node_base_y, color="black", marker='x')         # blue
    # ax1.scatter(node_base_xo, node_base_yo, color="green", marker='.')       # ? FAIL
    # ax2.scatter(node_sbj_xo, node_sbj_yo, color="red", marker='.')          # ? FAIL
    # ax2.scatter(node_sbj_x, node_sbj_y, color="black", marker='x')
    # plt.show()
    #
    subject_mesh_nodes = np.hstack((node_sbj_x[:, None], node_sbj_y[:, None]))
    # print(subject_mesh_nodes)

    # Plot the morph
    ord = np.argsort(node_theta)
    xo_base = np.multiply(node_base_ro[ord], -np.cos(node_theta[ord]))
    yo_base = np.multiply(node_base_ro[ord], -np.sin(node_theta[ord]))
    xo_sbj = np.multiply(node_sbj_ro[ord], -np.cos(node_theta[ord]))
    yo_sbj = np.multiply(node_sbj_ro[ord], -np.sin(node_theta[ord]))

    if show_plots:
        plt.figure(figsize=(1000/96, 800/96), dpi=96)
        plt.scatter(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], color='blue', s=3)
        plt.plot(xo_base, yo_base, color='blue')
        plt.xlim([-175, 175])
        plt.ylim([-125, 125])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.figure(figsize=(1000/96, 800/96), dpi=96)
        plt.scatter(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], color='blue', s=3)
        plt.scatter(node_sbj_x, node_sbj_y, color='orange', s=3)
        plt.plot(xo_base, yo_base, color='blue')
        plt.plot(xo_sbj, yo_sbj, color='orange')
        plt.quiver(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1],
                   node_sbj_x - base_mesh_nodes[:, 0], node_sbj_y - base_mesh_nodes[:, 1],
                   angles='xy', scale_units='xy', scale=1)
        plt.xlim([-175, 175])
        plt.ylim([-125, 125])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.figure(figsize=(1000/96, 800/96), dpi=96)
        plt.scatter(node_sbj_x, node_sbj_y, color='orange', s=3)
        plt.plot(xo_sbj, yo_sbj, color='orange')
        plt.xlim([-175, 175])
        plt.ylim([-125, 125])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        plt.figure(figsize=(1000/96, 800/96), dpi=96)
        plt.scatter(node_sbj_x, node_sbj_y, color='orange', s=3)
        plt.plot(xo_sbj, yo_sbj, color='orange')
        plt.xlim([-175, 175])
        plt.ylim([-125, 125])
        plt.gca().set_aspect('equal', adjustable='box')

    return subject_mesh_nodes


def get_nodes_in_lung(lung_coordinates, subject_mesh_nodes, sbj_mesh_centre=None):
    """Determine if each node is within the specified lung"""

    if sbj_mesh_centre is not None:
        lung_coordinates -= sbj_mesh_centre
    lung_path = pth.Path(lung_coordinates)
    lung_nodes = lung_path.contains_points(subject_mesh_nodes)

    return lung_nodes.astype(int)


def generate_mode_lung_nodes():

    mode_lung_nodes = np.empty([n_nodes, n_modes])

    # Iterate until accepted samples = n_samples
    for mode in range(n_modes):

        print("Generating lung nodes: mode {:d} of {:d}".format(mode+1, n_modes), end="\n")

        # Retrieve splines for the mode
        Fs_mode = Fs_mean + mode_splines[:, mode]

        # Evaluate the sampled parameterised shape
        eval_shapes = [[] for _ in bodies]
        for body in bodies:
            eval_shape_body = np.empty([0, len(eval_phis[body])])
            for dim in body_dims[body]:
                body_dim_indices = np.multiply((Is_body == body), (Js_body == dim))
                Fs_body_dim = Fs_mode[body_dim_indices]
                Ks_body_dim = Ks_mean[body_dim_indices]
                eval_shape = bsparam.evaluate_mesh_bspline(Fs_body_dim, Ks_body_dim, eval_phis[body])
                # print(body, dim)
                # print(np.shape(eval_shape))
                eval_shape_body = np.vstack([eval_shape_body, eval_shape])
            # print(np.shape(eval_shape_body))
            eval_shapes[body] = eval_shape_body
            # print(eval_shapes[body])

        # Generate subject specific mesh
        torso_indices = Is_body==2
        Fs_mode_torso = Fs_mode[torso_indices]
        Ks_mean_torso = Ks_mean[torso_indices]
        subject_mesh_nodes = distort_mesh(Fs_mode_torso, Ks_mean_torso)

        # Determine if each node is within the lungs
        lung_r_coords = eval_shapes[0].T
        lung_l_coords = eval_shapes[1].T
        # print(lung_r_coords)
        # print(lung_l_coords)
        lung_nodes_r = get_nodes_in_lung(lung_r_coords, subject_mesh_nodes)
        lung_nodes_l = get_nodes_in_lung(lung_l_coords, subject_mesh_nodes)

        if show_plots:
            plt.plot(lung_r_coords[:, 0], lung_r_coords[:, 1], color='black')
            plt.plot(lung_l_coords[:, 0], lung_l_coords[:, 1], color='black')
            plot_l = subject_mesh_nodes[lung_nodes_l == 1, :]
            plot_r = subject_mesh_nodes[lung_nodes_r == 1, :]
            plt.scatter(plot_l[:, 0], plot_l[:, 1], color='green', s=8)
            plt.scatter(plot_r[:, 0], plot_r[:, 1], color='green', s=8)
            plt.show()
            plt.close('all')

        mode_lung_nodes[:, mode] = (lung_nodes_r + lung_nodes_l)

    return mode_lung_nodes


def main():
    """..."""

    # Bayesian approach
    mode_lung_nodes = generate_mode_lung_nodes()
    print(mode_lung_nodes)
    print(np.sum(mode_lung_nodes, axis=0))

    # Node based lung function: MEAN
    # ??? lung_nodes_mean = np.mean(mode_lung_nodes, axis=1)

    # Node based lung function: COVARIANCE
    n_mlr_modes = len(Gamma_wgivenp)
    # Generate full conditoinal covariance for weights
    Gamma_wgivenp_full = np.eye(len(mode_splines.T))
    Gamma_wgivenp_full[:n_mlr_modes, :n_mlr_modes] = Gamma_wgivenp
    Gamma_ngivenp = np.matmul(np.matmul(mode_lung_nodes, Gamma_wgivenp_full), mode_lung_nodes.T)
    print(Gamma_ngivenp)

    # # To save data...
    # np.savetxt("Geom\\pca_"+pca_id+"\\pca_mean_mesh-"+mesh_id+".csv", lung_nodes_mean, delimiter=",")
    np.savetxt("Geom\\pca_"+pca_id+"\\pca_condcov_mesh-"+mesh_id+".csv", Gamma_ngivenp, delimiter=",")

    # Plot results
    lung_nodes_variances = np.sqrt(np.diag(Gamma_ngivenp))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
    ax.plot_trisurf(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], lung_nodes_variances,
                    cmap='viridis', linewidth=0, antialiased=False)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
    ax.plot_trisurf(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], lung_nodes_variances,
                    cmap='viridis', linewidth=0, antialiased=False)
    ax.view_init(azim=90, elev=90)
    plt.show()


if __name__ == "__main__":
    main()
