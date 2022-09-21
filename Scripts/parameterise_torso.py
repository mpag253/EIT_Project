import numpy as np
from Slice2D import assembly_plane_intersection_curve as assmcurve
from Slice2D import read_in_data as read_in_data
import BSplineParameterisation.bspline_parameterisation as bsparam


def get_parameterised_curve(int_curve, nknots, convert_polar=False, convert_rho=False, discont_elems=None, plot_bsplines=False, plot_bases=False):
    knots = np.linspace(0, 2*np.pi, nknots+1)[1:]
    body_coords = int_curve[:, 2:4]
    [Fs, _] = bsparam.get_parameterised_curve(body_coords, knots, convert_polar=convert_polar, convert_rho=convert_rho,
                                              discont_elems=discont_elems, plot_bsplines=plot_bsplines, plot_bases=plot_bases)
    return [Fs, knots]

def parameterise_torso(path, node_filename, elem_filename, plane, nICpts, nknots, ignore_elems, plot_3d=True, plot_2d=True):

    # Read in element and node data for each body
    nData = read_in_data.from_exnode(path, [node_filename])  # acting as a template
    [eData, sData, aData] = read_in_data.from_exelem(path, [elem_filename])

    # Get the intersection curves for all bodies and elements
    int_curves = assmcurve.get_intersection_curves(eData, aData, nData, plane, nICpts, ignore_elems,
                                                   plot_3d=plot_3d, plot_2d=plot_2d)
    # print(int_curves_mean[0][:, 2:4])
    # np.savetxt("torso_mean_raw.csv", int_curves_mean[0][:, 2:4], delimiter=",")

    # Shift body to centre
    body_coords = int_curves[0][:, 2:5]
    centre = [int(0.5*(np.max(body_coords[:, 0]) + np.min(body_coords[:, 0]))),
              int(0.5*(np.max(body_coords[:, 1]) + np.min(body_coords[:, 1])))]
    # print("centre: ", centre)
    int_curve = int_curves[0]
    int_curve[:, :2] -= centre

    # Fit a b-spline to the intersection curve
    [Fs, knots] = get_parameterised_curve(int_curve, nknots, convert_polar=True, convert_rho=False,
                                              discont_elems=None, plot_bsplines=True, plot_bases=False)

    return [Fs, knots, centre, body_coords]