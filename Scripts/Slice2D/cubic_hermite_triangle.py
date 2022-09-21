import numpy as np
import warnings

##############################
# CURRENTLY NOT WORKING !!!
##############################


def get_shape_function_coefficients(apex_node):
    """Generate the coefficients of the 2D quadrilateral cubic Hermite element.

    Output
    :return: list of coefficients [zeta_1_coeff, zeta_2_coeff, zeta_3_coeff] for apex node 1 triangles or
             [eta_1_coeff, eta_2_coeff, eta_3_coeff] for apex node 2 triangles, where [zeta/eta]_X_coeff are the
             basis function coefficients of [xi^0 xi^1 xi^2 xi^3] for zeta/eta subscript X (e.g. (xi^2)*(xi-1)
             has the coefficients [0, 0, -1, 1]).
    """

    # Coefficients of the shape functions: "psi_XY_coeff" for psi subscript X, superscript Y
    psi_10_coeff = np.array([[1., 0., -3., 2.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_11_coeff = np.array([[0., 1., -2., 1.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_20_coeff = np.array([[0., 0., 3., -2.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
    psi_21_coeff = np.array([[0., 0., -1., 1.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]

    if apex_node == 1:
        # Coefficients of the shape functions: "zeta_X_coeff" for zeta subscript X
        zeta_1_coeff = np.array([[1., -2.,  1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        zeta_2_coeff = np.array([[0.,  2., -1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        zeta_3_coeff = np.array([[0., -1.,  1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        coeff_list = [psi_10_coeff, psi_11_coeff, psi_20_coeff, psi_21_coeff, zeta_1_coeff, zeta_2_coeff, zeta_3_coeff]

    elif apex_node == 3:
        # Coefficients of the shape functions: "zeta_X_coeff" for zeta subscript X
        eta_1_coeff = np.array([[1., 0., -1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        eta_2_coeff = np.array([[0., 0.,  1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        eta_3_coeff = np.array([[0., 1., -1., 0.]])  # basis function coefficients of [xi^0 xi^1 xi^2 xi^3]
        coeff_list = [psi_10_coeff, psi_11_coeff, psi_20_coeff, psi_21_coeff, eta_1_coeff, eta_2_coeff, eta_3_coeff]

    else:
        raise ValueError("Invalid apex node for collapsed element.")

    return coeff_list


def get_coords_from_xis(xis, nodes, apex_node):
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
    [cC10, cC11, cC20, cC21, cQ1, cQ2, cQ3] = get_shape_function_coefficients(apex_node)

    # Evaluates shape functions: "psiXY_Z" for psi subscript X, superscript Y, and xi subscript Z
    psi10_1 = np.dot(cC10[0], xi1_array)
    psi11_1 = np.dot(cC11[0], xi1_array)
    psi20_1 = np.dot(cC20[0], xi1_array)
    psi21_1 = np.dot(cC21[0], xi1_array)
    if apex_node == 1:
        zta1_2 = np.dot(cQ1[0], xi2_array)
        zta2_2 = np.dot(cQ2[0], xi2_array)
        zta3_2 = np.dot(cQ3[0], xi2_array)
    elif apex_node == 3:
        eta1_2 = np.dot(cQ1[0], xi2_array)
        eta2_2 = np.dot(cQ2[0], xi2_array)
        eta3_2 = np.dot(cQ3[0], xi2_array)

    # Evaluating the coordinates from the interpolation functions
    if apex_node == 1:
        x_coord = (zta1_2*zta1_2*n1[0] + psi10_1*zta2_2*n2[0] + psi20_1*zta2_2*n3[0] +
                                       + psi11_1*zta2_2*n2[1] + psi21_1*zta2_2*n3[1] +
                                       + psi10_1*zta3_2*n2[2] + psi20_1*zta3_2*n3[2] +
                                       + psi11_1*zta3_2*n2[3] + psi21_1*zta3_2*n3[3])
        y_coord = (zta1_2*zta1_2*n1[4] + psi10_1*zta2_2*n2[4] + psi20_1*zta2_2*n3[4] +
                                       + psi11_1*zta2_2*n2[5] + psi21_1*zta2_2*n3[5] +
                                       + psi10_1*zta3_2*n2[6] + psi20_1*zta3_2*n3[6] +
                                       + psi11_1*zta3_2*n2[7] + psi21_1*zta3_2*n3[7])
        z_coord = (zta1_2*zta1_2*n1[8] + psi10_1*zta2_2*n2[8] + psi20_1*zta2_2*n3[8] +
                                       + psi11_1*zta2_2*n2[9] + psi21_1*zta2_2*n3[9] +
                                       + psi10_1*zta3_2*n2[10] + psi20_1*zta3_2*n3[10] +
                                       + psi11_1*zta3_2*n2[11] + psi21_1*zta3_2*n3[11])
    if apex_node == 3:
        x_coord = (psi10_1*eta1_2*n1[0] + psi20_1*eta1_2*n2[0] + eta2_2*eta2_2*n3[0] +
                   psi11_1*eta1_2*n1[1] + psi21_1*eta1_2*n2[1] +
                   psi10_1*eta3_2*n1[2] + psi20_1*eta3_2*n2[2] +
                   psi11_1*eta3_2*n1[3] + psi21_1*eta3_2*n2[3])
        y_coord = (psi10_1*eta1_2*n1[4] + psi20_1*eta1_2*n2[4] + eta2_2*eta2_2*n3[4] +
                   psi11_1*eta1_2*n1[5] + psi21_1*eta1_2*n2[5] +
                   psi10_1*eta3_2*n1[6] + psi20_1*eta3_2*n2[6] +
                   psi11_1*eta3_2*n1[7] + psi21_1*eta3_2*n2[7])
        z_coord = (psi10_1*eta1_2*n1[8] + psi20_1*eta1_2*n2[8] + eta2_2*eta2_2*n3[8] +
                   psi11_1*eta1_2*n1[9] + psi21_1*eta1_2*n2[9] +
                   psi10_1*eta3_2*n1[10] + psi20_1*eta3_2*n2[10] +
                   psi11_1*eta3_2*n1[11] + psi21_1*eta3_2*n2[11])

    return [x_coord, y_coord, z_coord]


def get_whole_element(nPts, node_data, apex_node):
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
            [x[i, j], y[i, j], z[i, j]] = get_coords_from_xis([xi1s[i], xi2s[j]], node_data, apex_node)

    return [x, y, z]


def get_interp_matrices(nodes, apex_node):
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
    [cC10, cC11, cC20, cC21, cQ1, cQ2, cQ3] = get_shape_function_coefficients(apex_node)

    # Interpolation matrices for d = x, y, and z (sum of transpose(C)*A*node_value for each)
    sum_ctaf = np.zeros([3, 4, 4])
    for d in range(3):
        if apex_node == 1:
            # x_coord = (psi10_1*psi10_2*n1[0] + psi20_1*psi10_2*n2[0] + psi10_1*psi20_2*n3[0] + psi20_1*psi20_2*n4[0] +
            #            psi11_1*psi10_2*n1[1] + psi21_1*psi10_2*n2[1] + psi11_1*psi20_2*n3[1] + psi21_1*psi20_2*n4[1] +
            #            psi10_1*psi11_2*n1[2] + psi20_1*psi11_2*n2[2] + psi10_1*psi21_2*n3[2] + psi20_1*psi21_2*n4[2] +
            #            psi11_1*psi11_2*n1[3] + psi21_1*psi11_2*n2[3] + psi11_1*psi21_2*n3[3] + psi21_1*psi21_2*n4[3])
            # sum_ctaf[d, :, :] = (
            #         c10.T*c10*n1[d*4 + 0] + c10.T*c20*n2[d*4 + 0] + c20.T*c10*n3[d*4 + 0] + c20.T*c20*n4[d*4 + 0] +
            #         c10.T*c11*n1[d*4 + 1] + c10.T*c21*n2[d*4 + 1] + c20.T*c11*n3[d*4 + 1] + c20.T*c21*n4[d*4 + 1] +
            #         c11.T*c10*n1[d*4 + 2] + c11.T*c20*n2[d*4 + 2] + c21.T*c10*n3[d*4 + 2] + c21.T*c20*n4[d*4 + 2] +
            #         c11.T*c11*n1[d*4 + 3] + c11.T*c21*n2[d*4 + 3] + c21.T*c11*n3[d*4 + 3] + c21.T*c21*n4[d*4 + 3])
            # x_coord = (zta1_2*zta1_2*n1[0] + psi10_1*zta2_2*n2[0] + psi20_1*zta2_2*n3[0] +
            #                                + psi11_1*zta2_2*n2[1] + psi21_1*zta2_2*n3[1] +
            #                                + psi10_1*zta3_2*n2[2] + psi20_1*zta3_2*n3[2] +
            #                                + psi11_1*zta3_2*n2[3] + psi21_1*zta3_2*n3[3])
            sum_ctaf[d, :, :] = (cQ1.T*cQ1*n1[d*4 + 0] + cQ2.T*cC10*n2[d*4 + 0] + cQ2.T*cC20*n3[d*4 + 0] +
                                                       + cQ2.T*cC11*n2[d*4 + 1] + cQ2.T*cC21*n3[d*4 + 1] +
                                                       + cQ3.T*cC10*n2[d*4 + 2] + cQ3.T*cC20*n3[d*4 + 2] +
                                                       + cQ3.T*cC11*n2[d*4 + 3] + cQ3.T*cC21*n3[d*4 + 3])
        elif apex_node == 3:
            # x_coord = (psi10_1*eta1_2*n1[0] + psi20_1*eta1_2*n2[0] + eta2_2*eta2_2*n3[0] +
            #            psi11_1*eta1_2*n1[1] + psi21_1*eta1_2*n2[1] +
            #            psi10_1*eta3_2*n1[2] + psi20_1*eta3_2*n2[2] +
            #            psi11_1*eta3_2*n1[3] + psi21_1*eta3_2*n2[3])
            sum_ctaf[d, :, :] = (cQ1.T*cC10*n1[d*4 + 0] + cQ1.T*cC20*n2[d*4 + 0] + cQ2.T*cQ2*n3[d*4 + 0] +
                                 cQ1.T*cC11*n1[d*4 + 1] + cQ1.T*cC21*n2[d*4 + 1] +
                                 cQ3.T*cC10*n1[d*4 + 2] + cQ3.T*cC20*n2[d*4 + 2] +
                                 cQ3.T*cC11*n1[d*4 + 3] + cQ3.T*cC21*n2[d*4 + 3])

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


def get_intcurve_by_screening_xi1(nICpts, xi1_lims, sumCtAfs, nodes, plane, info, apex_node):
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

        # iterate through each root
        for j in range(len(xi_2_roots)):
            # test if the root is real, if so, get the coordinates
            if not (np.iscomplex(xi_2_roots[j])) and (0. <= xi_2_roots[j].real <= 1.):
                intCurve[i, :] = get_coords_from_xis([xi_1s[i], xi_2_roots[j].real], nodes, apex_node)
                # intCurve[i, :, j]
                validCount += 1
            # else:
            #     # if the root is complex, return NaN as the coords for that root
            #     intCurve[i, :, j] = [float("NaN"), float("NaN"), float("NaN")]


        if validCount > 1:
            # This warning will print if there are more than one valid root in the element domain
            print("WARNING: more than one valid solution of xi_2 - check solution carefully!\t(body: " +
                  str(int(body)) + ", element: " + str(int(elem)) + ", xi_1=" + "{:.4f}".format(xi_1s[i]) +
                  "..., xi_2_roots=(" + np.array2string(xi_2_roots) + ")). ")
            print("^ TEMP SOLUTION IN DEVELOPMENT")

    return intCurve


def get_intcurve_by_screening_xi2(nICpts, xi2_lims, sumCtAfs, nodes, plane, info, apex_node):
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

        for j in range(len(xi_1_roots)):
            if not (np.iscomplex(xi_1_roots[j])) and (0. <= xi_1_roots[j].real <= 1.):
                intCurve[i, :] = get_coords_from_xis([xi_1_roots[j].real, xi_2s[i]], nodes, apex_node)
                validCount += 1

        if validCount > 1:
            # This warning will print if there are more than one valid root in the element domain
            print("WARNING: more than one valid solution of xi_1 - check solution carefully!\t(body: " +
                  str(int(body)) + ", element: " + str(int(elem)) + ", xi_2=" + "{:.4f}".format(xi_2s[i]) +
                  "..., xi_1_roots=(" + np.array2string(xi_1_roots) + ")). ")

    return intCurve


def get_edge_intercepts(sum_ctafs, node_data, int_plane, apex_node):
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
    edge_intercepts = np.zeros([9, 3])

    # Find intercepts on xi_1 = 0 and 1
    for xi_1 in [0, 1]:
        xi_2_roots = get_xi2_roots(xi_1, sum_ctafs, int_plane)
        for j in range(len(xi_2_roots)):
            if ~np.iscomplex(xi_2_roots[j]) and (0. <= xi_2_roots[j].real <= 1.):
                coords = get_coords_from_xis([xi_1, xi_2_roots[j].real], node_data, apex_node)
                residual = get_point_plane_residual(coords, int_plane)
                if abs(residual) < 1e-6:
                    edge_intercepts[3*xi_1 + j, :] = coords
                # edge_intercepts[3*xi_1 + j, :] = coords

    # Find intercepts on xi_2 = 0 and 1
    for xi_2 in [0]:
        xi_1_roots = get_xi1_roots(xi_2, sum_ctafs, int_plane)
        for j in range(len(xi_1_roots)):
            if ~np.iscomplex(xi_1_roots[j]) and (0. <= xi_1_roots[j].real <= 1.):
                coords = get_coords_from_xis([xi_1_roots[j].real, xi_2], node_data, apex_node)
                residual = get_point_plane_residual(coords, int_plane)
                if abs(residual) < 1e-6:
                    edge_intercepts[3*(xi_2 + 2) + j, :] = coords
                # edge_intercepts[3*(xi_2 + 2) + j, :] = coords

    return edge_intercepts
