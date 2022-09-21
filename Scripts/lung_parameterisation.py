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


def pickle_dump(filename, data):
    # Save data to file (pickle)
    pkl_file = open(filename+'.pkl', "wb")
    pickle.dump(data, pkl_file)
    pkl_file.close()
    print("\nSaved data to pickle: ", filename+'.pkl')


def prepare_lung_fit_data(int_curve, dc_elems):

    # Test if the intersection curve is complete, raise error if not
    if not np.all(np.isclose(int_curve[0, 2:], int_curve[-1, 2:])):
        raise ValueError("Intersection incomplete for the body")

    # Reordering to start with discontinuity + repeat first point at end (non-periodic data) <---------------------------------- FIX?
    start_idx = 0
    for i in range(1, len(int_curve[:, 1])):
        e1 = int_curve[i, 1]
        e2 = int_curve[i-1, 1]
        if (e1 in dc_elems[0] and e2 in dc_elems[1]) or (e1 in dc_elems[1] and e2 in dc_elems[0]):
            start_idx = i
    if start_idx != 0:
        n_idx = np.shape(int_curve[:, 1])[0]
        sort_idxs = np.concatenate((np.arange(start_idx, n_idx), np.arange(start_idx + 1)))  # "start_idx+1" repeats first point
        int_curve = int_curve[sort_idxs, :]

    # Define x-y coordinates of the body
    body_coords = int_curve[:, [2, 3]]

    # # Find centre and shift shape to centre
    # centre = (np.max(body_coords[:, 0:2], axis=0) + np.min(body_coords[:, 0:2], axis=0))/2
    # body_coords[:, 0:2] = body_coords[:, 0:2] - [centre]

    # Using normalised, cumulative arclength as parameter
    arclens = np.sqrt(np.sum(np.square(body_coords[1:, :] - body_coords[:-1, :]), axis=1))
    cumperim = np.cumsum(arclens)
    cumperim_norm = cumperim/cumperim[-1]
    cumperim_norm = np.concatenate(([0], cumperim_norm))
    fitdata_x = np.concatenate(([cumperim_norm], [body_coords[:, 0]]), axis=0).T
    fitdata_y = np.concatenate(([cumperim_norm], [body_coords[:, 1]]), axis=0).T

    # # Plot shape
    # npts = np.shape(int_curve[:, 1])[0]
    # cum_elem_counts = np.empty([npts, 1])
    # for i, j in enumerate(int_curve[:, 1]):
    #     cum_elem_counts[i] = np.count_nonzero(int_curve[0:i + 1, 1] == j)
    # first_elems = [i for i, j in enumerate(cum_elem_counts) if j == 1]
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(body_coords[:, 0], body_coords[:, 1])  #, marker='x', color="blue", linestyle='None')
    # plt.plot(body_coords[first_elems, 0], body_coords[first_elems, 1], marker='.', color="red", linestyle='None')
    # ax.set_aspect('equal')

    return [fitdata_x, fitdata_y]


def get_lung_param_fits(fit_data, nknots):

    [fit_data_x, fit_data_y] = fit_data

    # # Plot shape
    # # (for plotting out elements, doesn't run correctly without?? -- no idea)
    # npts = np.shape(int_curve[:, 1])[0]
    # cum_elem_counts = np.empty([npts, 1])
    # for i, j in enumerate(int_curve[:, 1]):
    #     cum_elem_counts[i] = np.count_nonzero(int_curve[0:i + 1, 1] == j)
    # first_elems = [i for i, j in enumerate(cum_elem_counts) if j == 1]
    #
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(fit_data_x[:, 0], fit_data_x[:, 1])  #, marker='x', color="blue", linestyle='None')
    # plt.plot(fit_data_x[first_elems, 0], fit_data_x[first_elems, 1],
    #          marker='.', color="red", linestyle='None')
    # #ax.set_aspect('equal')
    #
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(fit_data_y[:, 0], fit_data_y[:, 1])  #, marker='x', color="blue", linestyle='None')
    # plt.plot(fit_data_y[first_elems, 0], fit_data_y[first_elems, 1],
    #          marker='.', color="red", linestyle='None')
    # #ax.set_aspect('equal')
    #
    # plt.draw()
    # plt.show()
    # # plt.close('all')

    # Define mesh of knots for the NON-periodic b-spline
    # mesh = np.linspace(0*np.pi, 2*np.pi, 51)[0:-1]

    # Define mesh for the b-spline
    knots = np.linspace(0, 1, nknots)
    # knots = np.concatenate(([knots[0]], [knots[0]], knots, [knots[-1]], [knots[-1]]))
    knots = np.concatenate(([knots[0]], knots, [knots[-1]]))

    # Generate the B-spline for the intersection curve
    # [Fs_x, _] = bsparam.fit_mesh_bspline_to_data(fit_data_x, mesh, discontinuous_elements=None, plot_bases=False)
    # [Fs_y, _] = bsparam.fit_mesh_bspline_to_data(fit_data_y, mesh, discontinuous_elements=None, plot_bases=False)
    [Fs_x, _] = bsparam.get_parameterised_curve(fit_data_x, knots, convert_polar=False, convert_rho=False,
                                                discont_elems=None, plot_bsplines=True, plot_bases=False)
    [Fs_y, _] = bsparam.get_parameterised_curve(fit_data_y, knots, convert_polar=False, convert_rho=False,
                                                discont_elems=None, plot_bsplines=True, plot_bases=False)

    # print(Fs_x, Fs_y)

    # Collate data for output
    param_data = np.hstack((np.transpose([knots]), Fs_x[0], Fs_y[0]))
    # param_data = np.hstack((np.transpose([mesh]), Fs_x, Fs_y))

    # plt.show()

    return param_data


def get_parameterised_curves(int_curves, nknots, discont_elems, other):

    # Curve Parameterisation
    try:
        lung_fit_data = prepare_lung_fit_data(int_curves[0], discont_elems)
        print(lung_fit_data)
    except ValueError as e:
        if str(e) == "Intersection incomplete for the body":
            print("\nREJECTED SAMPLE: An intersection curve was incomplete")
            pickle_dump('incomplete_int_curve', [int_curves, other])
            param_data = []
        else:
            print('other error')
            raise
    else:
        try:
            param_data = get_lung_param_fits(lung_fit_data, nknots)
        except np.linalg.LinAlgError as e:
            if str(e) == "SVD did not converge in Linear Least Squares":
                print("\nREJECTED SAMPLE: Linear least squares B-spline fit failed")
                param_data = []
            else:
                print('other error')
                raise

    return param_data


def get_evaluated_bspline(param_data, evalmesh):

    # Evaluate fits
    # evalmesh = np.linspace(0., 1., 1000)
    eval_x = bsparam.evaluate_mesh_bspline(param_data[:, 1], param_data[:, 0], evalmesh)
    eval_y = bsparam.evaluate_mesh_bspline(param_data[:, 2], param_data[:, 0], evalmesh)
    return np.vstack((eval_x, eval_y)).T


def parameterise_lung(path, node_filename, elem_filename, plane, nICpts=100, nknots=25, ignore_elems=[], discont_elems=[[], []]):
    """ ... """

    # Read in element and node data for each body
    nData = read_in_data.from_exnode(path, [node_filename])
    [eData, sData, aData] = read_in_data.from_exelem(path, [elem_filename])

    # Get the intersection curves for all bodies and elements
    int_curves_lung = assmcurve.get_intersection_curves(eData, aData, nData, plane, nICpts, ignore_elems,
                                                         plot_3d=True, plot_2d=True)

    # Parameterise with b-spline and evaluate
    param_data_lung = get_parameterised_curves(int_curves_lung, nknots, discont_elems, [eData, nData])
    print(param_data_lung)

    #Plot
    # evalmesh = np.linspace(0., 1., 1000)
    # get_evaluated_bspline(param_data_lung, evalmesh)

    return param_data_lung


def main():

    # INPUTS ###########################################################################################################

    path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

    # Define the filenames of the ".exnode" and ".exelem" files for each body
    nFile = 'TEST_population_sample_mean'
    lung = 'Left'
    node_filename = "Geom\\" + nFile + "\\fitted" + lung + ".exnode"
    elem_filename = "Geom\\template\\template" + lung + ".exelem"

    # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
    plane = [0., 0., 1., 167.]  # [0., 0., 1., 120.]

    # Define elements to ignore (e.g. lobe elements)
    ignore_elems = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 111, 112, 113, 114, 115, 117]
    # Define discontinuity elements in each body
    # [[elements on one side of discont.],[elements on other side of discont.]]
    if lung == 'Right':
        discont_elems = [[35, 27, 18, 7], [31, 23, 13, 1]]
    elif lung == 'Left':
        discont_elems = [[98, 97, 89, 80, 69], [94, 93, 85, 75, 63]]
    else:
        discont_elems = [[], []]

    nICpts = 100
    nknots = 25

    # ##################################################################################################################

    parameterise_lung(path, node_filename, elem_filename, plane, nICpts, nknots, ignore_elems, discont_elems)

    # ##################################################################################################################


if __name__ == "__main__":
    main()