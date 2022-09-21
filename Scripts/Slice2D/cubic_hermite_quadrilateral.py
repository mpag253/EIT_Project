import numpy as np
import warnings


def get_shape_function_coefficients():
    """Generate the coefficients of the 2D quadrilateral cubic Hermite element.

    Output
    :return: list of coefficients [psi_10_coeff, psi_11_coeff, psi_20_coeff, psi_21_coeff], where psi_XY_coeff are the
             basis function coefficients of [xi^0 xi^1 xi^2 xi^3] for psi subscript X, superscript Y (e.g. (xi^2)*(xi-1)
             has the coefficients [0, 0, -1, 1]).
    """

    # Coefficients of the shape functions: "psi_XY_coeff" for psi subscript X, superscript Y
    psi_10_coeff = np.array([[1., 0., -3., 2.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_11_coeff = np.array([[0., 1., -2., 1.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_20_coeff = np.array([[0., 0., 3., -2.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_21_coeff = np.array([[0., 0., -1., 1.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]

    return [psi_10_coeff, psi_11_coeff, psi_20_coeff, psi_21_coeff]


def get_coords_from_xis(xis, nodes):
    """Returns the global coordinates (x, y, z) at a given local coordinate (xi-space).

    Input
    :param xis: list [xi1, xi2] containing the values of xi_1 and xi_2 at which to evaluate the global coordinates.
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.

    Output
    :return: list of global coordinates (x, y, z).
    """

    [xi1, xi2] = xis
    [n1, n2, n3, n4] = nodes

    # Arrays of xi^i for i = [0, 1, 2, 3]
    xi1_array = np.array([1., xi1, xi1 ** 2., xi1 ** 3.])
    xi2_array = np.array([1., xi2, xi2 ** 2., xi2 ** 3.])

    # Coefficients of the shape functions (psi): "cXY" for psi subscript X, superscript Y
    [c10, c11, c20, c21] = get_shape_function_coefficients()

    # Evaluates shape functions: "psiXY_Z" for psi subscript X, superscript Y, and xi subscript Z
    psi10_1 = np.dot(c10[0], xi1_array)
    psi11_1 = np.dot(c11[0], xi1_array)
    psi20_1 = np.dot(c20[0], xi1_array)
    psi21_1 = np.dot(c21[0], xi1_array)
    psi10_2 = np.dot(c10[0], xi2_array)
    psi11_2 = np.dot(c11[0], xi2_array)
    psi20_2 = np.dot(c20[0], xi2_array)
    psi21_2 = np.dot(c21[0], xi2_array)

    # Evaluating the coordinates from the interpolation functions
    x_coord = (psi10_1*psi10_2*n1[0] + psi20_1*psi10_2*n2[0] + psi10_1*psi20_2*n3[0] + psi20_1*psi20_2*n4[0] +
               psi11_1*psi10_2*n1[1] + psi21_1*psi10_2*n2[1] + psi11_1*psi20_2*n3[1] + psi21_1*psi20_2*n4[1] +
               psi10_1*psi11_2*n1[2] + psi20_1*psi11_2*n2[2] + psi10_1*psi21_2*n3[2] + psi20_1*psi21_2*n4[2] +
               psi11_1*psi11_2*n1[3] + psi21_1*psi11_2*n2[3] + psi11_1*psi21_2*n3[3] + psi21_1*psi21_2*n4[3])
    y_coord = (psi10_1*psi10_2*n1[4] + psi20_1*psi10_2*n2[4] + psi10_1*psi20_2*n3[4] + psi20_1*psi20_2*n4[4] +
               psi11_1*psi10_2*n1[5] + psi21_1*psi10_2*n2[5] + psi11_1*psi20_2*n3[5] + psi21_1*psi20_2*n4[5] +
               psi10_1*psi11_2*n1[6] + psi20_1*psi11_2*n2[6] + psi10_1*psi21_2*n3[6] + psi20_1*psi21_2*n4[6] +
               psi11_1*psi11_2*n1[7] + psi21_1*psi11_2*n2[7] + psi11_1*psi21_2*n3[7] + psi21_1*psi21_2*n4[7])
    z_coord = (psi10_1*psi10_2*n1[8] + psi20_1*psi10_2*n2[8] + psi10_1*psi20_2*n3[8] + psi20_1*psi20_2*n4[8] +
               psi11_1*psi10_2*n1[9] + psi21_1*psi10_2*n2[9] + psi11_1*psi20_2*n3[9] + psi21_1*psi20_2*n4[9] +
               psi10_1*psi11_2*n1[10] + psi20_1*psi11_2*n2[10] + psi10_1*psi21_2*n3[10] + psi20_1*psi21_2*n4[10] +
               psi11_1*psi11_2*n1[11] + psi21_1*psi11_2*n2[11] + psi11_1*psi21_2*n3[11] + psi21_1*psi21_2*n4[11])

    return [x_coord, y_coord, z_coord]


def get_whole_element(nPts, node_data):
    """Generates a mesh of global coordinates (x, y, z) that represent the bicubic Hermite element.

    Input
    :param nPts: number of local coordinate points (xi-space) in each direction to evaluate the global coordinates.
    :param node_data: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                      [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                      node X.

    Output
    :return: list [x, y, z] of matrices where each element is the corresponding global coordinate of a mesh point.
    """

    # Define arrays of xi_1 and xi_2 to evaluate the element coordinates
    xi1s = np.linspace(0., 1., nPts)
    xi2s = np.linspace(0., 1., nPts)

    # Preallocate the coordinate arrays
    x = np.zeros([nPts, nPts])
    y = np.zeros([nPts, nPts])
    z = np.zeros([nPts, nPts])

    # Iterate through all combinations of xi_1 and xi_2 to evaluate coordinates
    for i in range(len(xi1s)):
        for j in range(len(xi2s)):
            [x[i, j], y[i, j], z[i, j]] = get_coords_from_xis([xi1s[i], xi2s[j]], node_data)

    return [x, y, z]


def get_interp_matrices(nodes):
    """Generate the interpolation matrices 'sum_CtAf' in the x-, y- and z-directions. These matrices act on
    [xi_1^0, xi_1^1, xi_1^2, xi_1^3] and [xi_2^0, xi_2^1, xi_2^2, xi_2^3] to give the corresponding global coordinates
    (x, y, z).

    Input
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.

    Output
    :return: interpolation matrices 'sum_CtAf', the sum of [[the product of the basis function coefficients (C and A)]
             multiplied by the respective nodal variable (f)]
    """

    [n1, n2, n3, n4] = nodes

    # Coefficients of the shape functions (psi): "cXY" for psi subscript X, superscript Y
    [c10, c11, c20, c21] = get_shape_function_coefficients()

    # Interpolation matrices for d = x, y, and z (sum of transpose(C)*A*node_value for each)
    sum_ctaf = np.zeros([3, 4, 4])
    for d in range(3):
        sum_ctaf[d, :, :] = (
                    c10.T*c10*n1[d*4 + 0] + c10.T*c20*n2[d*4 + 0] + c20.T*c10*n3[d*4 + 0] + c20.T*c20*n4[d*4 + 0] +
                    c10.T*c11*n1[d*4 + 1] + c10.T*c21*n2[d*4 + 1] + c20.T*c11*n3[d*4 + 1] + c20.T*c21*n4[d*4 + 1] +
                    c11.T*c10*n1[d*4 + 2] + c11.T*c20*n2[d*4 + 2] + c21.T*c10*n3[d*4 + 2] + c21.T*c20*n4[d*4 + 2] +
                    c11.T*c11*n1[d*4 + 3] + c11.T*c21*n2[d*4 + 3] + c21.T*c11*n3[d*4 + 3] + c21.T*c21*n4[d*4 + 3])

    return sum_ctaf


def get_xi2_roots(xi1, sumCtAfs, int_plane):
    """Get the values of xi_2 that lie on an intersection plane for a given value of xi_1. Note: the xi_2 are not
     necessarily real or within the element domain.

    Input
    :param xi1: the value of xi_1 at which to evaluate xi_2.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param int_plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: the values (roots) of xi_2 that are the intersection of the given xi_1 and the intersection plane.
    """

    [xsumCtAf, ysumCtAf, zsumCtAf] = sumCtAfs

    # Arrays of (xi_1)^i for i = [0, 1, 2, 3]
    xi1_array = np.array([[1., xi1, xi1 ** 2., xi1 ** 3.]]).T

    # Evaluate the polynomial coefficients of xi2_array for the intersection with the plane
    xi2_coeffs = ((np.dot(xsumCtAf, xi1_array).T)*int_plane[0] +
                  (np.dot(ysumCtAf, xi1_array).T)*int_plane[1] +
                  (np.dot(zsumCtAf, xi1_array).T)*int_plane[2] +
                  + np.array([[1, 0, 0, 0]])*int_plane[3])

    # Solve for the roots of xi_2 that satisfy the plane equation
    xi2_roots = np.roots(np.flip(xi2_coeffs[0]))

    return xi2_roots


def get_xi1_roots(xi2, sumCtAfs, int_plane):
    """Get the values of xi_1 that lie on an intersection plane for a given value of xi_2. Note: the xi_1 are not
     necessarily real or within the element domain.

    Input
    :param xi2: the value of xi_1 at which to evaluate xi_1.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param int_plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: the values (roots) of xi_1 that are the intersection of the given xi_2 and the intersection plane.
    """

    [xsumCtAf, ysumCtAf, zsumCtAf] = sumCtAfs

    # Arrays of (xi_1)^i for i = [0, 1, 2, 3]
    xi2_array = np.array([[1., xi2, xi2 ** 2., xi2 ** 3.]]).T

    # Evaluate the polynomial coefficients of xi1_array for the intersection with the plane
    xi1_coeffs = ((np.dot(xi2_array.T, xsumCtAf))*int_plane[0] +
                  (np.dot(xi2_array.T, ysumCtAf))*int_plane[1] +
                  (np.dot(xi2_array.T, zsumCtAf))*int_plane[2] +
                  + np.array([[1, 0, 0, 0]])*int_plane[3])

    # Solve for the roots of xi_1 that satisfy the plane equation
    xi1_roots = np.roots(np.flip(xi1_coeffs[0]))

    return xi1_roots


def get_point_plane_residual(point, plane):
    residual = np.sum(np.dot(point, plane[0:3])) + plane[3]
    return residual


def get_intcurve_by_screening_xi1(nICpts, xi1_lims, sumCtAfs, nodes, plane, info):
    """Generate the coordinates of an intersection curve by assuming values of xi_1 and solving for the roots of xi_2.

    Input
    :param nICpts: number of intersection curve points to evaluate on the domain.
    :param xi1_lims: the limits of the domain of xi_1 at which to evaluate xi_2.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.
    :param plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: global coordinates of the intersection curve, rows [x, y, z] for all valid points.
    """

    [body, elem] = info

    # Defining an array of xi_1 values at which to evaluate xi_2
    xi_1s = np.linspace(xi1_lims[0], xi1_lims[1], nICpts)

    # Iterate through all values of xi_1 and retain the intersection point coordinates if they are valid within the
    # domain of the element
    # intCurve = np.zeros([nICpts, 3, 3])
    intCurve = np.zeros([nICpts, 3])

    for i in range(len(xi_1s)):
        xi_2_roots = get_xi2_roots(xi_1s[i], sumCtAfs, plane)
        validCount = 0

        # remove any duplicate roots
        for j in range(1, len(xi_2_roots)):
            for k in range(len(xi_2_roots[0:j])):
                if xi_2_roots[j] == xi_2_roots[k]:
                    xi_2_roots[j] = np.inf
                    # print("Found duplicate root and set it to Inf.")

        # iterate through each root
        for j in range(len(xi_2_roots)):
            # test if the root is real, if so, get the coordinates
            if not (np.iscomplex(xi_2_roots[j])) and (0. <= xi_2_roots[j].real <= 1.):
                intCurve[i, :] = get_coords_from_xis([xi_1s[i], xi_2_roots[j].real], nodes)
                # intCurve[i, :, j]
        #         validCount += 1
        #     # else:
        #     #     # if the root is complex, return NaN as the coords for that root
        #     #     intCurve[i, :, j] = [float("NaN"), float("NaN"), float("NaN")]
        #
        # if validCount > 1:
        #     # This warning will print if there are more than one valid root in the element domain
        #     print("WARNING: more than one valid solution of xi_2 - check solution carefully!\t(body: " +
        #           str(int(body)) + ", element: " + str(int(elem)) + ", xi_1=" + "{:.4f}".format(xi_1s[i]) +
        #           "..., xi_2_roots=(" + np.array2string(xi_2_roots) + ")). ")

    return intCurve


def get_intcurve_by_screening_xi2(nICpts, xi2_lims, sumCtAfs, nodes, plane, info):
    """Generate the coordinates of an intersection curve by assuming values of xi_2 and solving for the roots of xi_1.

    Input
    :param nICpts: number of intersection curve points to evaluate on the domain.
    :param xi2_lims: the limits of the domain of xi_2 at which to evaluate xi_1.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.
    :param plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: global coordinates of the intersection curve, rows [x, y, z] for all valid points.
    """

    [body, elem] = info

    # Defining an array of xi_1 values at which to evaluate xi_2
    xi_2s = np.linspace(xi2_lims[0], xi2_lims[1], nICpts)

    # Iterate through all values of xi_2 and retain the intersection point coordinates if they are valid within the
    # domain of the element
    intCurve = np.zeros([nICpts, 3])

    for i in range(len(xi_2s)):
        xi_1_roots = get_xi1_roots(xi_2s[i], sumCtAfs, plane)
        validCount = 0

        # remove any duplicate roots
        for j in range(1, len(xi_1_roots)):
            for k in range(len(xi_1_roots[0:j])):
                if xi_1_roots[j] == xi_1_roots[k]:
                    xi_1_roots[j] = np.inf
                    # print("Found duplicate root and set it to Inf.")

        for j in range(len(xi_1_roots)):
            if not (np.iscomplex(xi_1_roots[j])) and (0. <= xi_1_roots[j].real <= 1.):
                intCurve[i, :] = get_coords_from_xis([xi_1_roots[j].real, xi_2s[i]], nodes)
        #         validCount += 1
        #
        # if validCount > 1:
        #     # This warning will print if there are more than one valid root in the element domain
        #     print("WARNING: more than one valid solution of xi_1 - check solution carefully!\t(body: " +
        #           str(int(body)) + ", element: " + str(int(elem)) + ", xi_2=" + "{:.4f}".format(xi_2s[i]) +
        #           "..., xi_1_roots=(" + np.array2string(xi_1_roots) + ")). ")

    return intCurve


def get_multi_intcurves_by_screening_xi1(nICpts, xi1_lims, edge_pts, sumCtAfs, nodes, plane, info):
    """Generate the coordinates of multiple intersection curves by assuming values of xi_1 and solving for the roots of
       xi_2.

    Input
    :param nICpts: number of intersection curve points to evaluate on the domain.
    :param xi1_lims: the limits of the domain of xi_1 at which to evaluate xi_2.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.
    :param plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: global coordinates of the intersection curve, rows [x, y, z] for all valid points.
    """

    [body, elem] = info

    # Defining an array of xi_1 values at which to evaluate xi_2
    xi_1s = np.linspace(xi1_lims[0], xi1_lims[1], nICpts)

    # Iterate through all values of xi_1 and retain the intersection point coordinates if they are valid within the
    # domain of the element
    # intCurve = np.zeros([nICpts, 3, 3])
    intCurve = np.zeros([nICpts, 3, 3])
    intCurve[:] = np.nan

    for i in range(len(xi_1s)):
        xi_2_roots = get_xi2_roots(xi_1s[i], sumCtAfs, plane)
        validCount = 0

        # remove any duplicate roots
        for j in range(1, len(xi_2_roots)):
            for k in range(len(xi_2_roots[0:j])):
                if xi_2_roots[j] == xi_2_roots[k]:
                    xi_2_roots[j] = np.inf
                    # print("Found duplicate root and set it to Inf.")

        # iterate through each root
        for j in range(len(xi_2_roots)):  # range(3)
            # test if the root is real, if so, get the coordinates
            if not (np.iscomplex(xi_2_roots[j])) and (0. <= xi_2_roots[j].real <= 1.):
                intCurve[i, :, j] = get_coords_from_xis([xi_1s[i], xi_2_roots[j].real], nodes)
        #         validCount += 1
        #     # else:
        #     #     # if the root is complex, return NaN as the coords for that root
        #     #     intCurve[i, :, j] = [float("NaN"), float("NaN"), float("NaN")]
        # if validCount > 1:
        #     # This warning will print if there are more than one valid root in the element domain
        #     print("WARNING: more than one valid solution of xi_2 - check solution carefully!\t(body: " +
        #           str(int(body)) + ", element: " + str(int(elem)) + ", xi_1=" + "{:.4f}".format(xi_1s[i]) +
        #           "..., xi_2_roots=(" + np.array2string(xi_2_roots) + ")). ")

    intCurves = get_organised_subcurves(intCurve, edge_pts)  # <-------------------------------------------------------


    # intCurves = [[], [], []]
    # for j in range(3):
    #     nan_idxs = np.isnan(intCurve[:, 0, j])
    #     intCurves[j] = intCurve[nan_idxs, :, j]

    return intCurves


def get_multi_intcurves_by_screening_xi2(nICpts, xi2_lims, edge_pts, sumCtAfs, nodes, plane, info):
    """Generate the coordinates of an intersection curve by assuming values of xi_2 and solving for the roots of xi_1.

    Input
    :param nICpts: number of intersection curve points to evaluate on the domain.
    :param xi2_lims: the limits of the domain of xi_2 at which to evaluate xi_1.
    :param sumCtAfs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param nodes: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                  [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                  node X.
    :param plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)

    Output
    :return: global coordinates of the intersection curve, rows [x, y, z] for all valid points.
    """

    [body, elem] = info

    # Defining an array of xi_1 values at which to evaluate xi_2
    xi_2s = np.linspace(xi2_lims[0], xi2_lims[1], nICpts)

    # Iterate through all values of xi_2 and retain the intersection point coordinates if they are valid within the
    # domain of the element
    intCurve = np.zeros([nICpts, 3, 3])
    intCurve[:] = np.nan

    for i in range(len(xi_2s)):
        xi_1_roots = get_xi1_roots(xi_2s[i], sumCtAfs, plane)
        validCount = 0

        # remove any duplicate roots
        for j in range(1, len(xi_1_roots)):
            for k in range(len(xi_1_roots[0:j])):
                if xi_1_roots[j] == xi_1_roots[k]:
                    xi_1_roots[j] = np.inf
                    print("Found duplicate root and set it to Inf.")

        for j in range(len(xi_1_roots)):
            if not (np.iscomplex(xi_1_roots[j])) and (0. <= xi_1_roots[j].real <= 1.):
                intCurve[i, :, j] = get_coords_from_xis([xi_1_roots[j].real, xi_2s[i]], nodes)
                # validCount += 1
            # else:
            #     # if the root is complex, return NaN as the coords for that root
            #     intCurve[i, :, j] = [float("NaN"), float("NaN"), float("NaN")]
        # if validCount > 1:
        #     # This warning will print if there are more than one valid root in the element domain
        #     print("WARNING: more than one valid solution of xi_1 - check solution carefully!\t(body: " +
        #           str(int(body)) + ", element: " + str(int(elem)) + ", xi_2=" + "{:.4f}".format(xi_2s[i]) +
        #           "..., xi_1_roots=(" + np.array2string(xi_1_roots) + ")). ")

    intCurves = get_organised_subcurves(intCurve, edge_pts)

    # intCurves = [[], [], []]
    # for j in range(3):
    #     nan_idxs = np.isnan(intCurve[:, 0, j])
    #     intCurves[j] = intCurve[nan_idxs, :, j]

    return intCurves


def get_organised_subcurves(intCurve, edge_pts):
    # Assemble all of the element subcurves into a single list
    subcurves = []
    for j in range(3):  # iterate through each root
        # find clusters of valid points in that root and add to the list
        root_subcurves = [intCurve[s, :, j] for s in np.ma.clump_unmasked(np.ma.masked_invalid(intCurve[:, 0, j]))]
        subcurves.extend(root_subcurves)
    # print("Subcurves raw:\n", subcurves)

    # Arrange the subcurves
    n_elem_curves = int(len(edge_pts)/2)  # maybe pass this as argument
    new_intCurves = [[] for _ in range(n_elem_curves)]
    unused_subcurves = np.ones(len(subcurves), dtype=np.int8)
    unused_edgepts = np.ones(len(edge_pts), dtype=np.int8)
    for n in range(n_elem_curves):
        start_edge_pt_idx = np.where(unused_edgepts == 1)[0][0]
        new_intCurves[n] = edge_pts[start_edge_pt_idx]
        unused_edgepts[start_edge_pt_idx] = 0  # remove point from usable points
        curve_latest_pt = edge_pts[start_edge_pt_idx]
        curve_complete = False
        while not curve_complete:
            # find closest end of all unused subcurves AND edgepoints
            min_dist_c = 1e12
            min_dist_e = 1e12
            for c in range(len(subcurves)):
                if unused_subcurves[c] == 1:
                    for end_idx in [0, -1]:
                        dist = np.sqrt(np.sum(np.square(curve_latest_pt - subcurves[c][end_idx, :])))
                        if dist < min_dist_c:
                            min_dist_c = dist
                            closest_pt_c = c
                            closest_pt_dir = end_idx
            for e in range(len(edge_pts)):
                if unused_edgepts[e] == 1:
                    dist = np.sqrt(np.sum(np.square(curve_latest_pt - edge_pts[e])))
                    if dist < min_dist_e:
                        min_dist_e = dist
                        closest_pt_e = e
            # use the closest point to append that subcurve
            # check if it was an edgepoint or subcurve
            if min_dist_e < min_dist_c:
                new_intCurves[n] = np.vstack((new_intCurves[n], edge_pts[closest_pt_e]))
                curve_latest_pt = new_intCurves[n][-1, :]
                unused_edgepts[closest_pt_e] = 0  # remove point from usable points
                curve_complete = True
            else:
                if closest_pt_dir == -1:  # flip the direction of the curve if last point was closest
                    subcurves[closest_pt_c] = np.flipud(subcurves[closest_pt_c])
                new_intCurves[n] = np.vstack((new_intCurves[n], subcurves[closest_pt_c]))
                curve_latest_pt = new_intCurves[n][-1, :]
                unused_subcurves[closest_pt_c] = 0  # remove point from usable points

    return new_intCurves

def get_edge_intercepts(sum_ctafs, node_data, int_plane):
    """Find the locations (global coordinates) where the plane intersects the edges of the element.

    Input
    :param sum_ctafs: interpolation matrices 'sum_CtAf' in the x-, y- and z-directions.
    :param node_data: node data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
                      [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at
                      node X.
    :param int_plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0).

    Output
    :return: numpy array of edge intercept global coordinates (size = 12 possible intercepts x 3 coordinate directions).
    """

    # Preallocate output: coordinates of edge intercepts ((4 edges * 3 roots) x (3 coordinate dims))
    edge_intercepts = np.zeros([12, 3])

    # Find intercepts on xi_1 = 0 and 1
    for xi_1 in [0, 1]:
        xi_2_roots = get_xi2_roots(xi_1, sum_ctafs, int_plane)

        # remove any duplicate roots
        for j in range(1, len(xi_2_roots)):
            for k in range(len(xi_2_roots[0:j])):
                if xi_2_roots[j] == xi_2_roots[k]:
                    xi_2_roots[j] = np.inf
                    # print("Found duplicate root and set it to Inf.")

        for j in range(len(xi_2_roots)):
            if ~np.iscomplex(xi_2_roots[j]) and (0. <= xi_2_roots[j].real <= 1.):
                coords = get_coords_from_xis([xi_1, xi_2_roots[j].real], node_data)
                residual = get_point_plane_residual(coords, int_plane)
                if abs(residual) < 1e-6:
                    edge_intercepts[3*xi_1 + j, :] = coords
                # edge_intercepts[3*xi_1 + j, :] = coords

    # Find intercepts on xi_2 = 0 and 1
    for xi_2 in [0, 1]:
        xi_1_roots = get_xi1_roots(xi_2, sum_ctafs, int_plane)

        # remove any duplicate roots
        for j in range(1, len(xi_1_roots)):
            for k in range(len(xi_1_roots[0:j])):
                if xi_1_roots[j] == xi_1_roots[k]:
                    xi_1_roots[j] = np.inf
                    # print("Found duplicate root and set it to Inf.")

        for j in range(len(xi_1_roots)):
            if ~np.iscomplex(xi_1_roots[j]) and (0. <= xi_1_roots[j].real <= 1.):
                coords = get_coords_from_xis([xi_1_roots[j].real, xi_2], node_data)
                residual = get_point_plane_residual(coords, int_plane)
                if abs(residual) < 1e-6:
                    edge_intercepts[3*(xi_2 + 2) + j, :] = coords
                # edge_intercepts[3*(xi_2 + 2) + j, :] = coords

    return edge_intercepts
