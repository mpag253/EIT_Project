import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import Slice2D.element_plane_intersection_curve as elemcurve
import Slice2D.assembly_plane_intersection_curve as assmcurve
import Slice2D.read_in_data as read_in_data
import BSplineParameterisation.bspline_parameterisation as bsparam
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


# def convert_intcurves_to_plane_local(global_coords, plane):
#     # Transform global 3D coordinates into local 2D coordinates of a plane
#     # see: https://stackoverflow.com/questions/49769459/convert-points-on-a-3d-plane-to-2d-coordinates
#
#     # Unpack plane coeffcients
#     [A, B, C, D] = plane
#
#     # Vectors and unit vectors of plane coordinate system
#     z_prime_v = np.array([A, B, C])
#     z_prime_uv = z_prime_v/np.sqrt(np.sum(z_prime_v**2))
#     if B != 0:
#         y_prime_v = np.array([0, C/B, 1.])
#     else:
#         y_prime_v = np.array([0, 1., 0])
#     y_prime_uv = y_prime_v/np.sqrt(np.sum(y_prime_v**2))
#     x_prime_v = np.cross(y_prime_uv, z_prime_uv)
#     x_prime_uv = x_prime_v/np.sqrt(np.sum(x_prime_v ** 2))
#
#     # Basis points
#     # origin-prime is the closest point on the plane to the origin
#     o_prime_bp = -D/(A**2 + B**2 + C**2)*np.array([A, B, C])
#     x_prime_bp = o_prime_bp + x_prime_uv
#     y_prime_bp = o_prime_bp + y_prime_uv
#     z_prime_bp = o_prime_bp + z_prime_uv
#
#     # Get transform matrix (M) by solving at basis points
#     S = np.concatenate((np.array([x_prime_bp, y_prime_bp, z_prime_bp, o_prime_bp]).T, np.ones([1, 4])))
#     Dmat = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [1., 1., 1., 1.]])
#     M = np.matmul(Dmat, np.linalg.inv(S))
#
#     # Do transform
#     local_coords = {}  # pre-allocate
#     max_oop = 0 # out-of-plane sum to check results
#     for b in range(len(global_coords)):
#         local_coords[b] = np.empty([np.shape(global_coords[b])[0], 4])  # pre-allocate
#         for p in range(np.shape(global_coords[b])[0]):
#             prep = np.transpose([np.concatenate((global_coords[b][p, 2:], [1.]))])  # prep vector for affine transform
#             local_coords[b][p, :] = np.matmul(M, prep).T[0]  # apply transform
#         max_oop += max(max_oop, np.max(np.absolute(local_coords[b][:, 2])))
#
#     tol = 1E-6
#     if max_oop < tol:
#         outdata = [local_coords[b][:, :2] for b in range(len(local_coords))]
#         return outdata
#     else:
#         print("Error in conversion to local coordinates: max out-of-plane value greater than tolerance.")
#         return []
#
#     # # Plotting and validation...
#     # fig = plt.figure()
#     # ax = plt.axes(projection='3d', proj_type = 'ortho')
#     # ax.plot(int_curves[0][:, 2], int_curves[0][:, 3], int_curves[0][:, 4])
#     # ax.plot(int_curves[1][:, 2], int_curves[1][:, 3], int_curves[1][:, 4])
#     # assmcurve.set_axes_equal(ax)
#     # ax.view_init(elev=69.5, azim=90)
#     # # ax.view_init(elev=0, azim=0)
#     #
#     # # Check the result manually for plane = [0, 0.5, 1, 10]
#     # theta = np.absolute(np.arctan(plane[1]/plane[2]))
#     # print(theta)
#     # o_dist = plane[3]*np.cos(theta)
#     # check_coords_y = -o_dist*np.sin(theta) + local_coords[0][:, 1]*np.cos(theta) + local_coords[0][:, 2]*np.sin(theta)
#     # check_coords_z = -o_dist*np.cos(theta) + local_coords[0][:, 1]*np.sin(theta) + local_coords[0][:, 2]*np.cos(theta)
#     # diff_y = check_coords_y - int_curves[0][:, 3]
#     # diff_z = check_coords_z - int_curves[0][:, 4]
#     # print(np.min(diff_y), np.max(diff_y))
#     # print(np.min(diff_z), np.max(diff_z))
#
#     return local_coords


def main():
    """Generate the intersection curve of the bicubic Hermite shape model and plane as defined in the INPUTS.
    """

    # INPUTS ###########################################################################################################

    path = "C:\\Users\\mipag_000\\Documents\\EIT_Project\\Slice2D\\Data\\"

    # Define the filenames of the ".exnode" and ".exelem" files for each body
    node_filenames = ["AGING006_Left_fitted.exnode", "AGING006_Right_fitted.exnode", "AGING006_Torso_fitted.exnode"] #["fittedLeft.exnode", "fittedRight.exnode"]
    elem_filenames = ["AGING006_Left_fitted.exelem", "AGING006_Right_fitted.exelem", "AGING006_Torso_fitted.exelem"] #["fittedLeft.exelem", "fittedRight.exelem"]

    # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
    plane = [0., 0., 1., 167.] #[0., 0., 1., 120.]
    # plane = [1., 0., 0., -170.]
    # plane = [0., .5, 1., 10.]

    # Define elements to ignore (e.g. lobe elements). This list may be incomplete.
    ignore_elems = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 111, 112, 113, 114, 115, 116, 117]

    # Define whether to plot outputs
    show_3d_plot = True
    show_2d_plot = True

    # ##################################################################################################################

    # Read in element and node data for each body
    nData = read_in_data.from_exnode(path, node_filenames)
    [eData, sData] = read_in_data.from_exelem(path, elem_filenames)

    # Get the intersection curves for all bodies and elements
    int_curves = assmcurve.get_intersection_curves(eData, nData, plane, ignore_elems,
                                         plot_3d=show_3d_plot, plot_2d=show_2d_plot)
    # print(int_curves[0][:, 2:])
    # np.savetxt('example_lung_geometry_left.csv', int_curves[0], delimiter=',')
    # np.savetxt('example_lung_geometry_rght.csv', int_curves[1], delimiter=',')
    plt.figure()
    ax = plt.axes(projection=None)
    plt.plot(int_curves[0][:, 2], int_curves[0][:, 3])
    plt.plot(int_curves[1][:, 2], int_curves[1][:, 3])
    plt.plot(int_curves[2][:, 2], int_curves[2][:, 3])
    ax.set_aspect('equal')

    # # Transform int_curve into local 2D (plane reference frame)
    # global_coords = [int_curves[b][:, 2:] for b in range(len(int_curves))]
    # local_coords = assmcurve.convert_intcurves_to_plane_local(global_coords, plane)
    # print(local_coords[0])
    # # Plot final shapes
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(local_coords[0][:, 0], local_coords[0][:, 1])
    # plt.plot(local_coords[1][:, 0], local_coords[1][:, 1])
    # ax.set_aspect('equal')
    plt.show()

    # (Optional) Write out data
    # print("Node data:\n", nData, "\n")
    # print("Element data:\n", eData, "\n")
    # print("Scale factors:\n", sData, "\n")
    # print("Full, organised intersection curve:\n", int_curves, "\n")

    # ##################################################################################################################

    # # Curve Parameterisation
    # # Get torso curve [NOT in polar coordinates] (USING LEFT LUNG TEMP !!!)
    # body_idx = 0
    # body_coords = int_curves[body_idx][:, [2, 3, 1]]
    # # find centroid and shift shape to centre
    # # centroid = np.mean(body_coords[:, 0:2], axis=0)
    # centre = (np.max(body_coords[:, 0:2], axis=0) + np.min(body_coords[:, 0:2], axis=0))/2
    # shift1 = [0, 0] #[200, 0] # [-25, 20]
    # body_coords[:, 0:2] = body_coords[:, 0:2] - [centre] + shift1
    # # transform lungs in x-direction to capture geometry more reliably
    # # print("body_coords:\n", body_coords)
    # # xpower = .5
    # # body_coords[:, 0] = np.power(np.abs(body_coords[:, 0]), xpower) * np.sign(body_coords[:, 0])
    # # body_coords[:, 0] = body_coords[:, 0] - shift1[0]**xpower
    # # xfactor = np.max(body_coords[:, 1]) / np.max(body_coords[:, 0])
    # # body_coords[:, 0] = body_coords[:, 0] * xfactor
    # # shift2 = [np.min(body_coords[:, 0])*.9, 0]
    # # body_coords[:, 0:2] = body_coords[:, 0:2] + [shift2]
    # # body_coords[:, 1] = np.sqrt(np.abs(body_coords[:, 1]))*np.sign(body_coords[:, 1])
    # body_theta = np.arctan2(body_coords[:, 1], body_coords[:, 0])
    # #print(body_theta)
    # # shift2 = [0.9*np.interp(-1/6*np.pi, body_theta, body_coords[:, 0], period=2*np.pi),
    # #           0.9*np.interp(-1/6*np.pi, body_theta, body_coords[:, 1], period=2*np.pi)]
    # # print(shift2)
    # # body_coords[:, 0:2] = body_coords[:, 0:2] - [shift2]
    #
    # npts = np.shape(int_curves[body_idx][:, 1])[0]
    # cum_elem_counts = np.empty([npts, 1])
    # for i, j in enumerate(int_curves[body_idx][:, 1]):
    #     cum_elem_counts[i] = np.count_nonzero(int_curves[body_idx][1:i+1, 1] == j)
    # first_elems = [i for i, j in enumerate(cum_elem_counts) if j == 1]
    #
    # fitdata1 = np.concatenate(([np.arange(npts)], [body_coords[:, 0]]), axis=0).T
    # fitdata2 = np.concatenate(([np.arange(npts)], [body_coords[:, 1]]), axis=0).T
    #
    # # Plot shape
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(body_coords[:, 0], body_coords[:, 1])
    # plt.plot(body_coords[first_elems, 0], body_coords[first_elems, 1], marker='.', color="red", linestyle = 'None')
    # ax.set_aspect('equal')
    #
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(body_coords[:, 0])
    # ax.set_aspect('equal')
    #
    # plt.figure()
    # ax = plt.axes(projection=None)
    # plt.plot(body_coords[:, 1])
    # ax.set_aspect('equal')
    #
    # plt.show()
    #
    #
    # # Define mesh of knots for the periodic b-spline
    # #mesh = np.linspace(0*np.pi, 2*np.pi, 51)[0:-1]
    # mesh = np.linspace(0, npts, 31)
    # mesh = np.concatenate(([mesh[0]], [mesh[0]], mesh, [mesh[-1]], [mesh[-1]]))
    # print(mesh)
    # # Generate the B-spline for the intersection curve
    # #[Fs, mesh] = bsparam.get_parameterised_curve(body_coords, mesh, convert_to_rho=False, plot_bsplines=True, plot_bases=True)
    # [Fs1, _] = bsparam.get_parameterised_curve(fitdata1, mesh, convert_polar=False, convert_rho=False, plot_bsplines=True, plot_bases=False)
    # [Fs2, _] = bsparam.get_parameterised_curve(fitdata2, mesh, convert_polar=False, convert_rho=False, plot_bsplines=True, plot_bases=False)


if __name__ == "__main__":
    main()
