import numpy as np
import warnings
from . import cubic_hermite_quadrilateral as chq
from . import cubic_hermite_triangle as cht


def get_intcurve_quadrilateral(node_data, int_plane, nICpts, info):
    """Generate the intersection curve from the element node data and the defined intersection plane.

    Input
    :param node_data: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                      [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                      node X.
    :param int_plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0).

    Output
    :return: intersection curve coordinates (rows [x, y, z]) for each valid point.
    """

    # Get the interpolation matrices (sum of transpose(C)*A*(node value, F)) for x, y, and, z directions
    sumCtAfs = chq.get_interp_matrices(node_data)

    # Test where the edge intercepts are
    edge_intercepts = chq.get_edge_intercepts(sumCtAfs, node_data, int_plane)
    # with np.printoptions(threshold=np.inf):
    #     print(edge_intercepts)

    # Count the number of edge intersections: total, each local direction, and each edge
    nInts = np.sum(edge_intercepts.any(1))
    nInts_xi1 = np.sum(edge_intercepts[0:6, :].any(1))
    nInts_xi2 = np.sum(edge_intercepts[6:12, :].any(1))
    nInts_xi1_0 = np.sum(edge_intercepts[0:3, :].any(1))
    nInts_xi1_1 = np.sum(edge_intercepts[3:6, :].any(1))
    nInts_xi2_0 = np.sum(edge_intercepts[6:9, :].any(1))
    nInts_xi2_1 = np.sum(edge_intercepts[9:12, :].any(1))
    edge_pts = edge_intercepts[edge_intercepts.any(1), :]
    # print("Edge points:\n", edge_pts)
    # print("Total number of edge interceptions: " + str(nInts))
    # print("Interception counts:", nInts, " (", nInts_xi1, nInts_xi2, nInts_xi1_0, nInts_xi1_1, nInts_xi2_0, nInts_xi2_1, ")")
    # Interpolate the intersection curve based on the results of the edge intersections
    # (i.e. handle interpolation differently depending on where the edge intercepts are)

    # # Define number of interpolation points
    # nICpts = 100

    if nInts == 0:
        # Condition with no intercepts.. potentially interior intersection
        # print("No edge intercepts. Screening through xi_1 (2x refinement) and closing off any loops.")
        # int_curve = chq.get_intcurve_by_screening_xi1(2*nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
        # int_curve = np.vstack([int_curve, int_curve[1, :]])
        int_curves = chq.get_multi_intcurves_by_screening_xi1(2*nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)
        if len(int_curves) > 0:
            int_curves[0] = np.vstack([int_curves[0], int_curves[0][1, :]])

    elif ~(nInts%2 == 0):
        # Uneven number of intersections is impossible
        raise ValueError("Uneven number of element edge intercepts is invalid!!!")

    elif nInts == 2:

        # print("Edge intercepts = 2 ...")

        if nInts_xi1_0 == 1 and nInts_xi1_1 == 1:
            # Straight intersection, use roots of xi_2 by screening through xi_1
            # print("Straight intercept in xi1")
            #int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)

        elif nInts_xi2_0 == 1 and nInts_xi2_1 == 1:
            # Straight intersection, use roots of xi_1 by screening through xi_2
            # print("Straight intercept in xi2")
            # int_curve = chq.get_intcurve_by_screening_xi2(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi2(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)


        elif nInts_xi1 == 1 and nInts_xi2 == 1:
            # Diagonal intersection condition
            # print("Diagonal intercept. Screening through xi_1 and adding boundary point.")
            # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            #
            # # Find xi_2 boundary point
            # if nInts_xi2_0 == 1:
            #     extraPt = chq.get_intcurve_by_screening_xi2(1, [0, 0], sumCtAfs, node_data, int_plane, info)
            # else:
            #     extraPt = chq.get_intcurve_by_screening_xi2(1, [1, 1], sumCtAfs, node_data, int_plane, info)
            #
            # # Append boundary point to correct position on curve
            # if nInts_xi1_0 == 1:
            #     int_curve = np.vstack([int_curve, extraPt])
            # else:
            #     int_curve = np.vstack([extraPt, int_curve])

            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)

        else:
            # Both intersections are on one edge
            # print("Both intercepts on one edge.")
            if nInts_xi1_0 == 2 or nInts_xi1_1 == 2:
                # int_curve = chq.get_intcurve_by_screening_xi2(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
                # how to handle adding intercepts?
                int_curves = chq.get_multi_intcurves_by_screening_xi2(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)
            elif nInts_xi2_0 == 2 or nInts_xi2_1 == 2:
                # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
                # how to handle adding intercepts?
                int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)                                                                                        # <<----- developing

    elif nInts > 2:
        # print("Edge intercepts > 2 ...")
        if nInts_xi1_0 >= 2 or nInts_xi1_1 >= 2:
            # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi2(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)     # <<----- developing
        elif nInts_xi2_0 >= 2 or nInts_xi2_1 >= 2:
            # int_curve = chq.get_intcurve_by_screening_xi2(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)     # <<----- developing
        else:
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)  # <<----- developing

        # with np.printoptions(threshold=np.inf):
        #     print("Intersection curve:\n", int_curves)

    else:
        # Something went very wrong if we got to here
        raise ValueError("Something went very wrong :(")

    # original!
    # # Eliminate any unevaluated rows (zero-rows) in the intersection curve
    # int_curve = int_curve[~np.all(int_curve == 0, axis=1)]
    # # print("Intersection curve coordinates:")
    # # print(int_curve)

    # develop!
    # Eliminate any unevaluated rows (zero-rows) in the intersection curve
    for i in range(len(int_curves)):
        int_curves[i] = int_curves[i][~np.all(int_curves[i] == 0, axis=1)]
    # print("Intersection curve coordinates:")
    # print(int_curve)

    # # TEMPORARY: stitch the curves of one element together (for validation only)
    # int_curve = int_curves[0]
    # if len(int_curves)>1:
    #     int_curve = np.vstack((int_curve, int_curves[1]))
    # if len(int_curves)>2:
    #     int_curve = np.vstack((int_curve, int_curves[2]))

    return int_curves


def get_intcurve_triangle(node_data, apex_node, int_plane, nICpts, info):
    """Generate the intersection curve from the element node data and the defined intersection plane.
       NOTE: STILL USING QUADRILATERAL FORMUALTION

    Input
    :param node_data: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                      [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                      node X.
    :param int_plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0).

    Output
    :return: intersection curve coordinates (rows [x, y, z]) for each valid point.
    """

    # Get the interpolation matrices (sum of transpose(C)*A*(node value, F)) for x, y, and, z directions
    sumCtAfs = chq.get_interp_matrices(node_data)#, apex_node)

    # Test where the edge intercepts are
    edge_intercepts = chq.get_edge_intercepts(sumCtAfs, node_data, int_plane)#, apex_node)

    # Count the number of edge intersections: total, each local direction, and each edge
    nInts = np.sum(edge_intercepts.any(1))
    nInts_xi1 = np.sum(edge_intercepts[0:6, :].any(1))
    nInts_xi2 = np.sum(edge_intercepts[6:9, :].any(1))
    nInts_xi1_0 = np.sum(edge_intercepts[0:3, :].any(1))
    nInts_xi1_1 = np.sum(edge_intercepts[3:6, :].any(1))
    edge_pts = edge_intercepts[edge_intercepts.any(1), :]
    # print("Edge points:\n", edge_pts)
    # print("Total number of edge interceptions: " + str(nInts))
    # print("Interception counts:")
    # print(nInts_xi1,nInts_xi2,nInts_xi1_0,nInts_xi1_1,nInts_xi2_0,nInts_xi2_1)

    # Interpolate the intersection curve based on the results of the edge intersections
    # (i.e. handle interpolation differently depending on where the edge intercepts are)

    # Define number of interpolation points
    nICpts = 100

    if nInts == 0:
        # Condition with no intercepts.. potentially interior intersection
        # print("No edge intercepts. Screening through xi_1 (2x refinement) and closing off any loops.")
        # int_curve = chq.get_intcurve_by_screening_xi1(2*nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)#, apex_node)
        # int_curve = np.vstack([int_curve, int_curve[1, :]])
        int_curves = chq.get_multi_intcurves_by_screening_xi1(2*nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)
        if len(int_curves) > 0:
            int_curves[0] = np.vstack([int_curves[0], int_curves[0][1, :]])

    elif nInts == 2:

        # print("Edge intercepts = 2 ...")

        if nInts_xi2 == 0:
            # No intercepts on xi2 = 0 (and xi2 = 1 doesn't exist), use roots of xi_2 by screening through xi_1
            # print("No intercepts on xi2 = 0 (and xi2 = 1 doesn't exist)")
            # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)#, apex_node)
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)

        elif nInts_xi2 == 2:
            # Two intercepts on xi2 = 0 (and xi2 = 1 doesn't exist), use roots of xi_2 by screening through xi_1
            # print("Two intercepts on xi2 = 0 (and xi2 = 1 doesn't exist)")
            # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)#, apex_node)
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)

        else:
            #  Single intercept on xi2 = 0 (and xi2 = 1 doesn't exist), use roots of xi_1 by screening through xi_2
            # print("Single intercept on xi2 = 0 (and xi2 = 1 doesn't exist)")
            # int_curve = chq.get_intcurve_by_screening_xi2(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)#, apex_node)
            int_curves = chq.get_multi_intcurves_by_screening_xi2(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)

    elif nInts > 2:

        # print("Edge intercepts > 2 ...")

        # condition with multiple intersection curves - more complicated
        # short term solution would be to output all curve points and let them be assembled manually
        # int_curve = np.empty([0, 3])
        if nInts_xi1_0 >= 2 or nInts_xi1_1 >= 2:
            # int_curve = chq.get_intcurve_by_screening_xi1(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi2(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)     # <<----- developing
        elif nInts_xi2 >= 2:
            # int_curve = chq.get_intcurve_by_screening_xi2(nICpts, [0, 1], sumCtAfs, node_data, int_plane, info)
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)     # <<----- developing
        else:
            int_curves = chq.get_multi_intcurves_by_screening_xi1(nICpts, [0.0001, 0.9999], edge_pts, sumCtAfs, node_data, int_plane, info)  # <<----- developing

        # with np.printoptions(threshold=np.inf):
        #     print("Intersection curve:\n", int_curves)

    else:
        # Something went very wrong if we got to here
        raise ValueError("Something went very wrong :(")

    # original!
    # # Eliminate any unevaluated rows (zero-rows) in the intersection curve
    # int_curve = int_curve[~np.all(int_curve == 0, axis=1)]
    # # print("Intersection curve coordinates:")
    # # print(int_curve)

    # develop!
    # Eliminate any unevaluated rows (zero-rows) in the intersection curve
    for i in range(len(int_curves)):
        int_curves[i] = int_curves[i][~np.all(int_curves[i] == 0, axis=1)]
    # print("Intersection curve coordinates:")
    # print(int_curve)

    # # TEMPORARY: stitch the curves of one element together (for validation only)
    # int_curve = int_curves[0]
    # if len(int_curves)>1:
    #     int_curve = np.vstack((int_curve, int_curves[1]))
    # if len(int_curves)>2:
    #     int_curve = np.vstack((int_curve, int_curves[2]))


    return int_curves






def get_intersection_curve(node_data, apex_node, int_plane, nICpts, info):

    if apex_node == 0:
        int_curves = get_intcurve_quadrilateral(node_data, int_plane, nICpts, info)
    else:
        # int_curve = get_intcurve_quadrilateral(node_data, int_plane, info)
        int_curves = get_intcurve_triangle(node_data, apex_node, int_plane, nICpts, info)

    return int_curves

# def main():
#
#     # IMPORT NODE DATA FOR THE ELEMENT
#
#     # Open spreadsheet with an example element
#     path = "C:\\Users\\mipag_000\\Documents\\EIT_Project\\Slice2D\\"
#     fName = "example_bicubic_hermite_element.xlsx"
#     wb = openpyxl.load_workbook(path + fName)
#     ws = wb.active
#
#     # Retrieve the node data
#     nodeData = np.array([[i.value for i in j] for j in ws['D8':'G19']])
#     nodeData = [nodeData[:, 0], nodeData[:, 1], nodeData[:, 2], nodeData[:, 3]]
#
#     # REPRODUCE AND PLOT THE WHOLE ELEMENT
#
#     [ex, ey, ez] = get_whole_element(10, nodeData)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     theCM = cm.get_cmap()
#     theCM._init()
#     alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
#     theCM._lut[:-3, -1] = alphas
#     surf = ax.plot_surface(ex, ey, ez, cmap=theCM, linewidth=0, antialiased=False)
#
#     # INTERSECTION CURVE
#
#     # Define the plane: coefficients [A, B, C, D] for plane equation Ax + By + Cz + D = 0
#     # plane = [0., 0., 1., 90.]
#     plane = [1., 0., 0., -170.]
#     # plane = [1., -1., 0., 0.]
#
#     # Generate the intersection curve
#     intCurve = get_intersection_curve(nodeData, plane)
#
#     # Add curve to the plot and show
#     ax.plot3D(intCurve[:, 0], intCurve[:, 1], intCurve[:, 2])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     # plt.xlim([80, 200])
#     # plt.ylim([100, 300])
#     # ax.set_zlim(-140, 100)
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
