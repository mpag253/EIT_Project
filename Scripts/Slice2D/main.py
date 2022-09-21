import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import element_plane_intersection_curve as elemcurve
import assembly_plane_intersection_curve as assmcurve
import read_in_data
import intersection_curve_bspline_parameterisation as bsparam
import scipy.interpolate as ip


# def main_single_element():
#     """Old 'main' function that runs the intersection curve for a single element.
#     """
#
#     # IMPORT NODE DATA FOR THE ELEMENT
#     # Open spreadsheet with an example element
#     path = "C:\\Users\\mipag_000\\Documents\\EIT_Project\\Slice2D\\Data\\"
#     fName = "example_bicubic_hermite_element.xlsx"
#     wb = openpyxl.load_workbook(path + fName)
#     ws = wb.active
#     # Retrieve the node data
#     nodeData = np.array([[i.value for i in j] for j in ws['D8':'G19']])
#     nodeData = [nodeData[:, 0], nodeData[:, 1], nodeData[:, 2], nodeData[:, 3]]
#
#     # REPRODUCE AND PLOT THE WHOLE ELEMENT
#     [ex, ey, ez] = elemcurve.get_whole_element(10, nodeData)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     theCM = cm.get_cmap()
#     theCM._init()
#     alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
#     theCM._lut[:-3, -1] = alphas
#     surf = ax.plot_surface(ex, ey, ez, cmap=theCM, linewidth=0, antialiased=False)
#
#     # INTERSECTION CURVE
#     # Define the plane: coefficients [A, B, C, D] for plane equation Ax + By + Cz + D = 0
#     plane = [0., 0., 1., 90.]
#     # plane = [1., 0., 0., -170.]
#     # plane = [1., -1., 0., 0.]
#     # Generate the intersection curve
#     intCurve = elemcurve.get_intersection_curve(nodeData, plane)
#     # Add curve to the plot and show
#     ax.plot3D(intCurve[:, 0], intCurve[:, 1], intCurve[:, 2])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     # plt.xlim([80, 200])
#     # plt.ylim([100, 300])
#     # ax.set_zlim(-140, 100)
#     plt.show()


def main():
    """Generate the intersection curve of the bicubic Hermite shape model and plane as defined in the INPUTS.
    """

    # INPUTS ###########################################################################################################

    path = "C:\\Users\\mipag_000\\Documents\\EIT_Project\\Slice2D\\Data\\"

    # Define the filenames of the ".exnode" and ".exelem" files for each body
    node_filenames = ["fittedLeft.exnode", "fittedRight.exnode"]
    elem_filenames = ["fittedLeft.exelem", "fittedRight.exelem"]

    # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
    plane = [0., 0., 1., 120.]
    # plane = [1., 0., 0., -170.]
    # plane = [1., -1., 0., 0.]

    # Define elements to ignore (e.g. lobe elements). This list may be incomplete.
    ignore_elems = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117]

    # Define whether to plot outputs
    show_3d_plot = False
    show_2d_plot = False

    # ##################################################################################################################

    # Read in element and node data for each body
    nData = read_in_data.from_exnode(path, node_filenames)
    [eData, sData] = read_in_data.from_exelem(path, elem_filenames)

    # Get the intersection curves for all bodies and elements
    int_curves = assmcurve.get_intersection_curves(eData, nData, plane, ignore_elems,
                                         plot_3d=show_3d_plot, plot_2d=show_2d_plot)



    # (Optional) Write out data
    # print("Node data:\n", nData, "\n")
    # print("Element data:\n", eData, "\n")
    # print("Scale factors:\n", sData, "\n")
    # print("Full, organised intersection curve:\n", int_curves, "\n")

    # ##################################################################################################################

    # # Get torso curve in polar coordinates (USING LEFT LUNG TEMP !!!)
    # body_idx = 0
    # body_coords = int_curves[body_idx][:, [2, 3, 1]]
    # centroid = np.mean(body_coords[:, 0:2], axis=0)
    # body_coords[:, 0:2] = body_coords[:, 0:2] - [centroid] + [-25, 20]
    #
    # # Define mesh of knots for the periodic b-spline
    # mesh = np.linspace(0*np.pi, 2*np.pi, 50)[0:-1]
    #
    # # Generate the B-spline for the intersection curve
    # [Fs, mesh] = bsparam.get_parameterised_curve(body_coords, mesh, discont_elems=None, plot_bsplines=True, plot_bases=True)


if __name__ == "__main__":
    main()
