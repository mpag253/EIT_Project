#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.path as pth
#from matplotlib import cm
# import Slice2D.element_plane_intersection_curve as elemcurve
import Slice2D.assembly_plane_intersection_curve as assmcurve
import Slice2D.read_in_data as read_in_data
import BSplineParameterisation.bspline_parameterisation as bsparam
#import scipy.interpolate as ip
#import time
from sys import exit
import copy
import pickle
#from scipy import stats
from scipy import sparse
import scipy.io
from matplotlib import path as pth
import parameterise_torso as parat

# INPUTS ###############################################################################################################

path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

# # --- PREDICTED ---
nFile = 'TEST_HLA-H11303_predicted'
node_filenames = ["Geom\\"+nFile+"\\fittedRight.exnode",
                  "Geom\\"+nFile+"\\fittedLeft.exnode",
                  "Geom\\"+nFile+"\\fittedTorso.exnode"]
elem_filenames = ["Geom\\template\\templateRight.exelem",
                  "Geom\\template\\templateLeft.exelem",
                  "Geom\\template\\templateTorso.exelem"]

# Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
plane = [0., 0., 1., 167.]

# Define elements to ignore (e.g. lobe elements)
ignore_elems = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 111, 112, 113, 114, 115, 117]
# Define discontinuity elements in each body
# ( [[elements on one side of discont.],[elements on other side of discont.]] ) for each body
discont_elems = [[[35, 27, 18, 7], [31, 23, 13, 1]],
                 [[98, 97, 89, 80, 69], [94, 93, 85, 75, 63]],
                 [[], []]]
lung_idx_r = 0
lung_idx_l = 1
lung_indices = [lung_idx_r, lung_idx_l]
nICpts = 100

# Bayesian
n_samples = 10
nmesh = 25

# Mesh
base = 'TEST_population_sample_mean'
base_mesh_nodes = np.genfromtxt('Geom\\'+base+'\\trimesh_nodes_0010.csv', delimiter=',')
# base_mesh_tris = np.genfromtxt('Geom\\'+base+'\\trimesh_tris_0010.csv', delimiter=',')
with open('Geom\\'+base+'\\bspdata_torso.pkl', 'rb') as f:
    bsp_data = pickle.load(f)
base_mesh_bsp_knots = bsp_data['knots']
base_mesh_bsp_Fs = bsp_data['Fs']
base_mesh_centre = bsp_data['centre']


def pickle_dump(filename, data):
    # Save data to file (pickle)
    pkl_file = open(filename+'.pkl', "wb")
    pickle.dump(data, pkl_file)
    pkl_file.close()
    print("\nSaved data to pickle: ", filename+'.pkl')


def get_normrand_vector(nData):
    # list of nodes (excluding versions)
    node_ids = np.unique(np.floor(nData[:, 0]))
    # list of nodes (including versions)
    vers_idxs = [np.where(np.floor(nData[:, 0]) == n)[0].tolist() for n in node_ids]
    # random sample from normal dist
    rsamp = np.empty(np.shape(nData[:, 1:]))
    rsamp_nod = np.random.normal(0, 1, [np.shape(node_ids)[0], 12])
    # print(rsamp_nod)
    rsamp_nod[:, [3, 7, 11]] = 0  # dont sample second deriv
    # print(rsamp_nod)
    for i in range(len(node_ids)):  # same sample for all node versions
        idxs = vers_idxs[i]
        for idx in idxs:
            rsamp[idx, :] = rsamp_nod[i, :]
    rsamp = rsamp.flatten()
    return rsamp


def test_complete_intcurves(int_curves):
    """Test if the intersection curve is complete"""

    accepted = True
    for i in range(len(int_curves)):
        int_curve = int_curves[i]
        if not np.all(np.isclose(int_curve[0, 2:], int_curve[-1, 2:], atol=1e-01)):
            accepted = False

    return accepted


def distort_mesh(int_curves, sbj_mesh_centre):
    """distort the mesh nodes based on fitted and imported bsplines"""

    # fit bspline to subject torso
    body_coords = int_curves[2][:, 2:]
    body_coords[:, :2] -= sbj_mesh_centre
    # print(body_coords)
    plt.close('all')
    knots_sbj = np.linspace(0, 2*np.pi, 100)[1:]
    [Fs_sbj, _] = bsparam.get_parameterised_curve(body_coords, knots_sbj, convert_polar=True, convert_rho=False,
                                                  discont_elems=None, plot_bsplines=False, plot_bases=False)
    # TRY EXCEPT TO CATCH B-SPLINE FAIL -----------------------------------------------------------------------------
    # HOW TO CATCH POOR B-SPLINE FIT??? -----------------------------------------------------------------------------
    # angular position of each node
    node_theta = np.arctan2(base_mesh_nodes[:, 1], base_mesh_nodes[:, 0]) + np.pi
    # radius of each node
    node_base_r = np.sqrt(np.square(base_mesh_nodes[:, 1]) + np.square(base_mesh_nodes[:, 0]))
    # base torso radius at each node angular position
    node_base_ro = bsparam.evaluate_mesh_bspline(base_mesh_bsp_Fs[0], base_mesh_bsp_knots, node_theta)
    # sampled torso radius at each node angular position
    node_sbj_ro = bsparam.evaluate_mesh_bspline(Fs_sbj[0], knots_sbj, node_theta)
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
    # exit(0)

    # Plot the morph
    ord = np.argsort(node_theta)
    xo_base = np.multiply(node_base_ro[ord], -np.cos(node_theta[ord]))
    yo_base = np.multiply(node_base_ro[ord], -np.sin(node_theta[ord]))
    xo_sbj = np.multiply(node_sbj_ro[ord], -np.cos(node_theta[ord]))
    yo_sbj = np.multiply(node_sbj_ro[ord], -np.sin(node_theta[ord]))
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
               node_sbj_x-base_mesh_nodes[:, 0], node_sbj_y-base_mesh_nodes[:, 1],
               angles='xy', scale_units='xy' , scale=1)
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


def get_sbj_mesh_centre(int_curves):
    """Centre the intcurves relative to the torso"""

    torso_min_x = np.min(int_curves[2][:, 2])
    torso_max_x = np.max(int_curves[2][:, 2])
    torso_min_y = np.min(int_curves[2][:, 3])
    torso_max_y = np.max(int_curves[2][:, 3])
    centre_x = (torso_max_x + torso_min_x)/2.
    centre_y = (torso_max_y + torso_min_y)/2.

    return [centre_x, centre_y]


def get_nodes_in_lung(int_curves, sbj_mesh_centre, subject_mesh_nodes):
    """Determine if each node is within the lungs"""

    lung_nodes_pre = np.zeros(len(subject_mesh_nodes))
    for i in lung_indices:
        int_curve = int_curves[i]
        lung_path = pth.Path(int_curve[:, 2:4] - sbj_mesh_centre)
        lung_nodes_pre = np.logical_or(lung_nodes_pre, lung_path.contains_points(subject_mesh_nodes))

    return lung_nodes_pre.astype(int)


def get_nodes_in_lung_left(int_curves, sbj_mesh_centre, subject_mesh_nodes):
    """Determine if each node is within the left lung"""

    int_curve = int_curves[lung_idx_l]
    lung_path = pth.Path(int_curve[:, 2:4] - sbj_mesh_centre)
    lung_nodes_l = lung_path.contains_points(subject_mesh_nodes)

    return lung_nodes_l.astype(int)


def get_nodes_in_lung_right(int_curves, sbj_mesh_centre, subject_mesh_nodes):
    """Determine if each node is within the right lung"""

    int_curve = int_curves[lung_idx_r]
    lung_path = pth.Path(int_curve[:, 2:4] - sbj_mesh_centre)
    lung_nodes_r = lung_path.contains_points(subject_mesh_nodes)

    return lung_nodes_r.astype(int)


def generate_samples(eData, aData, nData, nSig, plane, n_samples, ignore_elems, discont_elems, plot_3d=False, plot_2d=False):

    # Setup
    accept_counter = 0
    total_counter = 0
    # lung_nodes = np.zeros((n_samples+2, len(base_mesh_nodes)))  # [[] for _ in range(n_samples)]
    lung_nodes_l = np.zeros((n_samples, len(base_mesh_nodes)))
    lung_nodes_r = np.zeros((n_samples, len(base_mesh_nodes)))

    # Iterate until accepted samples = n_samples
    while accept_counter < n_samples:

        # Generate random sample from normal distribution
        rsamp = get_normrand_vector(nData)

        # Cholesky factorisation of covariance matrix
        nL = np.linalg.cholesky(nSig)

        # Randomly sampled node data
        nData_samp = copy.deepcopy(nData)
        nData_samp[:, 1:] = nData_samp[:, 1:] + np.reshape(np.matmul(nL,rsamp), np.shape(nData[:, 1:]))

        # Get the intersection curves for all bodies and elements
        int_curves = assmcurve.get_intersection_curves(eData, aData, nData_samp, plane, nICpts, ignore_elems,
                                                       plot_3d=plot_3d, plot_2d=plot_2d)

        # Test if intersection curve is complete
        accepted = test_complete_intcurves(int_curves)

        # Centre the intersection curves (to keep mesh uniform)
        sbj_mesh_centre = get_sbj_mesh_centre(int_curves)

        # Generate subject specific mesh
        subject_mesh_nodes = distort_mesh(int_curves, sbj_mesh_centre)

        # Determine if each node is within the lungs
        # lung_nodes[accept_counter, :] = get_nodes_in_lung(int_curves, subject_mesh_nodes)
        lung_nodes_l[accept_counter, :] = get_nodes_in_lung_left(int_curves, sbj_mesh_centre, subject_mesh_nodes)
        lung_nodes_r[accept_counter, :] = get_nodes_in_lung_right(int_curves, sbj_mesh_centre, subject_mesh_nodes)

        plt.plot(int_curves[0][:, 2]-sbj_mesh_centre[0], int_curves[0][:, 3]-sbj_mesh_centre[1], color='black')
        plt.plot(int_curves[1][:, 2]-sbj_mesh_centre[0], int_curves[1][:, 3]-sbj_mesh_centre[1], color='black')
        plot_l = subject_mesh_nodes[lung_nodes_l[accept_counter] == 1, :]
        plot_r = subject_mesh_nodes[lung_nodes_r[accept_counter] == 1, :]
        plt.scatter(plot_l[:, 0], plot_l[:, 1], color='green', s=8)
        plt.scatter(plot_r[:, 0], plot_r[:, 1], color='green', s=8)

        plt.show()
        plt.close('all')

        total_counter += 1
        if accepted:
            accept_counter += 1
            print("\nACCEPTED SAMPLE: {} of {} ({} total attempts)".format(str(accept_counter), str(n_samples), str(total_counter)))
        else:
            print("\nREJECTED SAMPLE: Intersection curve incomplete ")
            pickle_dump('incomplete_int_curve', [int_curves, nData_samp])

    # Node based lung function: mean and covariance
    # add one instance of all 0s and one instance of all 1s to prevent zero-values in cov matrix
    # # Lungs together:
    # lung_nodes[n_samples, :] = np.zeros(len(base_mesh_nodes))
    # lung_nodes[n_samples+1, :] = np.ones(len(base_mesh_nodes))
    # lung_nmean = np.mean(lung_nodes, axis=0)
    # lung_ncov = np.cov(lung_nodes.T)
    # Lungs separate:
    # uniform_addition = np.array([[0], [1]])*np.ones([1, len(base_mesh_nodes)])
    uniform_addition = 0.5*np.ones([1, len(base_mesh_nodes)])
    lung_nodes = np.concatenate((lung_nodes_r, lung_nodes_l, uniform_addition), axis=0)
    lung_nmean = np.mean(lung_nodes, axis=0)
    lung_ncov = np.cov(lung_nodes.T)


    # # Save the samples parameters to file (pickle)
    # print("\nSaving sampled lung node mean + cov.")
    # pickle_dump('sampled_lung_nodes_DEVELOP', [lung_nmean, lung_ncov])

    return lung_nmean, lung_ncov


def main():
    """..."""

    # Read in element and node data for each body
    nData_mean = read_in_data.from_exnode(path, node_filenames)
    nData_mean = nData_mean[nData_mean[:, 0].argsort()]
    [eData, _, aData] = read_in_data.from_exelem(path, elem_filenames)

    # Load covariance matrix (Sigma)
    nSig = sparse.load_npz("Geom\\TEST_population_sample_mean\\example_data_cov.npz").toarray()
    nSig += np.diag(1e-6*np.ones(np.shape(nSig)[0]))

    # # Get the intersection curves for all bodies and elements
    # ic_mean_l = assmcurve.get_intersection_curves(eData, aData, nData_mean, plane, nICpts, ignore_elems,
    #                                                       plot_3d=False, plot_2d=False)

    # Bayesian approach
    lung_nmean, lung_ncov = generate_samples(eData, aData, nData_mean, nSig, plane, n_samples,
                                             ignore_elems, discont_elems, plot_3d=True, plot_2d=True)
    # # To open pickled data...
    # with open('sampled_lung_nodes_DEVELOP.pkl', "rb") as f:
    #     [lung_nmean, lung_ncov] = pickle.load(f)

    # # To save data...
    np.savetxt("Geom\\TEST_HLA-H11303_predicted\\TEST_lung_nodes_mean.csv", lung_nmean, delimiter=",")
    np.savetxt("Geom\\TEST_HLA-H11303_predicted\\TEST_lung_nodes_cov.csv", lung_ncov, delimiter=",")

    # Plot results
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
    ax.plot_trisurf(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], lung_nmean,
                    cmap='viridis', linewidth=0, antialiased=False)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
    ax.plot_trisurf(base_mesh_nodes[:, 0], base_mesh_nodes[:, 1], lung_nmean,
                    cmap='viridis', linewidth=0, antialiased=False)
    ax.view_init(azim=90, elev=90)
    plt.show()

if __name__ == "__main__":
    main()
