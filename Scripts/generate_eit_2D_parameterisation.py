import numpy as np
import matplotlib.pyplot as plt
# import Slice2D.element_plane_intersection_curve as elemcurve
import Slice2D.assembly_plane_intersection_curve as assmcurve
import Slice2D.read_in_data as read_in_data
import BSplineParameterisation.bspline_parameterisation as bsparam
import parameterise_torso as pt
import parameterise_lung as pl
import copy
import pickle
from scipy import stats
from matplotlib import path as pth


# INPUTS ###############################################################################################################

path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

# pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-A'

parameterise_ground_truth = False
parameterise_sample_mean = False
parameterise_predicted_mean = False
parameterise_shape_modes = True

# Get the subject ID for the Leave-Out-Out
subject_ids = {'A':'H5977', 'B':'AGING043', 'C':'H7395', 'D':'AGING014', 'E':'AGING053'}
if parameterise_predicted_mean or parameterise_ground_truth or parameterise_shape_modes:
    subject_id = subject_ids[pca_id.split('-')[-1]]

# Load in elements
elem_filenames = ["Geom\\template\\templateRight.exelem",
                  "Geom\\template\\templateLeft.exelem",
                  "Geom\\template\\templateTorso.exelem"]
[eData, _, aData] = read_in_data.from_exelem(path, elem_filenames)

# 2D SLICE
# Define elements to ignore (e.g. lobe elements)
ignore_elems = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 111, 112, 113, 114, 115, 117]
# Define discontinuity elements in each body
# ( [[elements on one side of discont.],[elements on other side of discont.]] ) for each body
discont_elems = [[[35, 27, 18, 7], [31, 23, 13, 1]],
                 [[98, 97, 89, 80, 69], [94, 93, 85, 75, 63]],
                 [[], []]]
nICpts = 100

# PARAMETERISATION
[lung_index_r, lung_index_l] = [0, 1]
lung_indices = [lung_index_r, lung_index_l]
# Number of b-spline knots
n_knots = [25, 25, 40]

# # Mesh
# base = 'TEST_population_sample_mean'
# base_mesh_nodes = np.genfromtxt('Geom\\' + base + '\\trimesh_nodes_0010.csv', delimiter=',')
# # base_mesh_tris = np.genfromtxt('Geom\\'+base+'\\trimesh_tris_0010.csv', delimiter=',')
# with open('Geom\\' + base + '\\bspdata_torso.pkl', 'rb') as f:
#     bsp_data = pickle.load(f)
# Ks_base_mesh = bsp_data['knots']
# Fs_base_mesh = bsp_data['Fs'][0]
# # base_mesh_centre = bsp_data['centre']


def get_mesh_centre(int_curves):
    """Centre the intcurves relative to the torso"""

    torso_min_x = np.min(int_curves[2][:, 2])
    torso_max_x = np.max(int_curves[2][:, 2])
    torso_min_y = np.min(int_curves[2][:, 3])
    torso_max_y = np.max(int_curves[2][:, 3])
    centre_x = (torso_max_x + torso_min_x)/2.
    centre_y = (torso_max_y + torso_min_y)/2.

    return [centre_x, centre_y]


def get_filenames(id, format):
    if format == 1:
        node_filenames = ["Geom\\pca_" + pca_id + "\\" + id + "\\geom\\fittedRight.exnode",
                          "Geom\\pca_" + pca_id + "\\" + id + "\\geom\\fittedLeft.exnode",
                          "Geom\\pca_" + pca_id + "\\" + id + "\\geom\\fittedTorso.exnode"]
    elif format == 2:
        node_filenames = ["Geom\\pca_" + pca_id + "\\" + id + "\\geom\\Right_fitted_tf.exnode",
                          "Geom\\pca_" + pca_id + "\\" + id + "\\geom\\Left_fitted_tf.exnode",
                          "Geom\\pca_" + pca_id + "\\" + id + "\\geom\\Torso_fitted_tf.exnode"]
    return node_filenames


def parameterise_bodies(int_curves, mean_mesh_centre):

    Fs_mean = np.empty([0])
    Ks_mean = np.empty([0])
    Is_body = np.empty([0])
    Js_body = np.empty([0])

    # Get parameterised shapes (b-spline)
    for body in range(3):

        # Define intersection curve for the body and centre
        ic_body = int_curves[body]
        ic_body[:, 2:4] -= mean_mesh_centre

        # Get parameterised shapes (b-spline) for each body
        if body in lung_indices:
            param_data = pl.get_parameterised_curve(ic_body, n_knots[body], discont_elems=discont_elems[body],
                                                    plot_bsplines=False)
            Fss = param_data[0]
            Ks = param_data[1]

            for j, Fs in enumerate(Fss):
                Fs_mean = np.hstack([Fs_mean, Fs[0].flatten()])
                Ks_mean = np.hstack([Ks_mean, Ks])
                Is_body = np.hstack([Is_body, body*np.ones(np.shape(Ks))])  # body indices
                Js_body = np.hstack([Js_body, (j + 1)*np.ones(np.shape(Ks))])  # dimension indices

        else:
            param_data = pt.get_parameterised_curve(ic_body, n_knots[body], convert_polar=True,
                                                    plot_bsplines=False)
            [Fs, Ks] = param_data[:2]
            Fs_mean = np.hstack([Fs_mean, Fs[0].flatten()])
            Ks_mean = np.hstack([Ks_mean, Ks])
            Is_body = np.hstack([Is_body, body*np.ones(np.shape(Ks))])  # body indices
            Js_body = np.hstack([Js_body, np.ones(np.shape(Ks))])  # dimension indices

    return Is_body, Js_body, Fs_mean, Ks_mean


def get_eit_z_slice(mean_nData):
    node_num_zmin = 100
    node_num_zmax = 196
    mesh_zmin = mean_nData[mean_nData[:, 0] == (node_num_zmin + 0.1), 9]
    mesh_zmax = mean_nData[mean_nData[:, 0] == (node_num_zmax + 0.1), 9]
    slice_z = -(mesh_zmin + 0.518*(mesh_zmax - mesh_zmin))
    return slice_z


def parameterise_mean(node_filenames):

    # Read in element and node data for each body
    mean_nData = read_in_data.from_exnode(path, node_filenames)
    mean_nData = mean_nData[mean_nData[:, 0].argsort()]

    # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
    slice_z = get_eit_z_slice(mean_nData)
    plane = [0., 0., 1., slice_z]

    # Get the intersection curves for all bodies and elements
    ic_mean = assmcurve.get_intersection_curves(eData, aData, mean_nData, plane, nICpts, ignore_elems,
                                                     plot_3d=False, plot_2d=False)
    # Centre the intersection curves (to keep mesh uniform)
    mean_mesh_centre = get_mesh_centre(ic_mean)

    # Parameterise the bodies
    Is_body, Js_body, Fs_mean, Ks_mean = parameterise_bodies(ic_mean, mean_mesh_centre)
    param_data = np.vstack([Is_body, Js_body, Fs_mean, Ks_mean]).T

    return param_data, mean_mesh_centre


def main():
    """..."""

    print('\nPCA: '+pca_id+'\n')

    #####################
    # GROUND TRUTH
    #####################
    if parameterise_ground_truth:
        print('\rGenerating ground truth B-splines...', end='')
        # Geometry files
        node_filenames = get_filenames('truth_'+subject_id, format=2)
        # Parameterise the mean shape
        param_data, mesh_centre = parameterise_mean(node_filenames)
        # Save results
        np.save('Geom\\pca_' + pca_id + '\\truth_'+subject_id+'\\truth_bsplines.npy', param_data)
        np.save('Geom\\pca_' + pca_id + '\\truth_'+subject_id+'\\truth_mesh_centre.npy', mesh_centre)
        print('\rGenerating ground truth B-splines... Done.\n')

    #####################
    # SAMPLE MEAN SHAPE
    #####################
    if parameterise_sample_mean:
        print('Generating sample mean shape B-splines...', end='')
        # Geometry files
        node_filenames = get_filenames('sample_mean', format=1)
        # Parameterise the mean shape
        param_data, mesh_centre = parameterise_mean(node_filenames)
        # Save results
        np.save('Geom\\pca_' + pca_id + '\\sample_mean\\sample_bsplines_mean.npy', param_data)
        np.save('Geom\\pca_' + pca_id + '\\sample_mean\\sample_mean_mesh_centre.npy', mesh_centre)
        print('\rGenerating sample mean shape B-splines... Done.\n')

    ########################
    # PREDICTED MEAN SHAPE
    ########################
    if parameterise_predicted_mean:
        print('Generating predicted shape B-splines...', end='')
        # Geometry files
        node_filenames = get_filenames('predicted_'+subject_id, format=1)
        # Parameterise the mean shape
        param_data, mesh_centre = parameterise_mean(node_filenames)
        # SAVE RESULTS
        np.save('Geom\\pca_' + pca_id + '\\predicted_'+subject_id+'\\predicted_bsplines_mean.npy', param_data)
        np.save('Geom\\pca_' + pca_id + '\\predicted_'+subject_id+'\\predicted_mean_mesh_centre.npy', mesh_centre)
        print('\rGenerating predicted shape B-splines... Done.\n')

    ####################
    # SHAPE MODES + COVARIANCE
    ####################
    if parameterise_shape_modes:

        # Specify whether to generate the modes (slow) or load them from previous run
        load_modes = True

        if not load_modes:
            print('Generating shape mode B-splines...')

            # Load all mode shapes
            mode_shapes = np.load("Geom\\pca_" + pca_id + "\\pca_modeshapes_" + pca_id + ".npy").T

            # Load the sample mean
            node_filenames = get_filenames('sample_mean', format=1)
            mean_nData = read_in_data.from_exnode(path, node_filenames)
            mean_param_data = np.load('Geom\\pca_' + pca_id + '\\sample_mean\\sample_bsplines_mean.npy')
            [Is_body, Js_body, Fs_sample_mean, Ks_sample_mean] = mean_param_data.T
            mean_mesh_centre = np.load('Geom\\pca_' + pca_id + '\\sample_mean\\sample_mean_mesh_centre.npy')

            # GENERATE THE PARAMETERISED SHAPES FOR EACH MODE
            mode_splines = np.empty([len(Fs_sample_mean), len(mode_shapes)])
            for i, mode_shape in enumerate(mode_shapes): #  <---
                print("\r\tMode shape:", i, end='')

                # Update node values
                mode_nData = copy.deepcopy(mean_nData)
                mode_nData[:, 1:] += mode_shape.reshape(np.shape(mode_nData[:, 1:]))

                # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
                slice_z = get_eit_z_slice(mode_nData)
                plane = [0., 0., 1., slice_z]

                # Get the intersection curves for all bodies and elements
                ics_mode = assmcurve.get_intersection_curves(eData, aData, mode_nData, plane, nICpts, ignore_elems,
                                                            plot_3d=False, plot_2d=False)

                param_data = parameterise_bodies(ics_mode, mean_mesh_centre)
                Fs_mode = param_data[2]  #[Is_body, Js_body, Fs_mean, Ks_mean]

                # Subtract the mean shape
                mode_splines[:, i] = Fs_mode - Fs_sample_mean

            np.save('Geom\\pca_' + pca_id + '\\pca_mode_bsplines.npy', mode_splines)
            print('\nGenerating shape mode B-splines... Done.\n')

        else:
            mode_splines = np.load('Geom\\pca_' + pca_id + '\\pca_mode_bsplines.npy')
            mean_param_data = np.load('Geom\\pca_' + pca_id + '\\sample_mean\\sample_bsplines_mean.npy')
            [Is_body, Js_body, Fs_sample_mean, Ks_sample_mean] = mean_param_data.T
            print('Loaded previously-generated shape mode B-splines.\n')


        # Load onditional covariance of MLR mode shapes
        Gamma_wgivenp = np.load("Geom\\pca_" + pca_id + "\\pca_condcov_wgivenp_" + pca_id + ".npy")

        # CALCULATE SAMPLE COVARIANCE
        Gamma_w = np.eye(len(mode_splines.T))
        print(len(mode_splines.T))
        exit(0)
        Gamma_b = np.matmul(np.matmul(mode_splines, Gamma_w), mode_splines.T)

        # CALCULATE MLR PREDICTED COVARIANCE
        n_mlr_modes = len(Gamma_wgivenp)
        # Generate full conditoinal covariance for weights
        Gamma_wgivenp_full = np.eye(len(mode_splines.T))
        Gamma_wgivenp_full[:n_mlr_modes, :n_mlr_modes] = Gamma_wgivenp
        # Calculte the b-spline conditional covariance
        Gamma_bgivenp = np.matmul(np.matmul(mode_splines, Gamma_wgivenp_full), mode_splines.T)

        # # SAVE RESULTS
        np.save('Geom\\pca_'+pca_id+'\\sample_mean\\sample_bsplines_condcov.npy', Gamma_b)
        np.save('Geom\\pca_'+pca_id+'\\predicted_'+subject_id+'\\predicted_bsplines_condcov.npy', Gamma_bgivenp)

        # DISPLAY RESULTS
        with np.printoptions(precision=2, suppress=True):
            print('\nGamma_b:\n', Gamma_b)
            print('\nMode splines:\n', mode_splines)
            print('\nGamma_b|p:\n', Gamma_bgivenp)
            print('\nVariance b|p:\n', np.sqrt(np.diag(Gamma_bgivenp)))

        # Plot shapes
        plt.figure()
        for i in range(len(mode_splines.T)):
            Fs_mode = Fs_sample_mean + mode_splines[:, i]
            for body in range(3):
                body_indices = Is_body == body
                if body in lung_indices:
                    eval_1 = np.linspace(0, 1, 100)
                    eval_2 = np.empty([len(eval_1), 2])
                    for j in range(2):
                        dim_indices = Js_body == (j + 1)
                        body_dim_indices = np.multiply(body_indices, dim_indices)
                        Fs_dim = Fs_mode[body_dim_indices]
                        Ks_dim = Ks_sample_mean[body_dim_indices]
                        eval_2[:, j] = bsparam.evaluate_mesh_bspline(Fs_dim, Ks_dim, eval_1)
                    eval_x = eval_2[:, 0]
                    eval_y = eval_2[:, 1]
                else:
                    eval_1 = np.linspace(0, 2*np.pi, 100)
                    Fs_dim = Fs_mode[body_indices]
                    Ks_dim = Ks_sample_mean[body_indices]
                    eval_2 = bsparam.evaluate_mesh_bspline(Fs_dim, Ks_dim, eval_1)
                    eval_x = np.multiply(eval_2, -np.cos(eval_1))
                    eval_y = np.multiply(eval_2, -np.sin(eval_1))

                # plt.arrow(eval_x[0], eval_y[0], eval_x[1] - eval_x[0], eval_y[1] - eval_y[0],
                #           color='black', width=1)
                plt.plot(eval_x, eval_y,c='red')
                plt.axis('equal')
        plt.show()



    return


if __name__ == "__main__":
    main()
