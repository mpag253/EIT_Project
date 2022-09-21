import numpy as np
import matplotlib.pyplot as plt
from . import element_plane_intersection_curve as elemcurve
from . import cubic_hermite_quadrilateral as chq

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    :param ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_organised_intersection_curves(int_curves, ignore_elems, plot_3d_axis="off", plot_2d_axis="off"):
    """Organises the raw intersection curves of an assortment of bodies and elements into ordered, continuous
    intersection curves for each body.

    Input
    :param int_curves: the raw intersection curves to be organised, columns are [body, element, x, y, z].
    :param ignore_elems: list of element numbers to ignore from the organised intersection curve.
    :param plot_3d_axis: "off" (default, to disable 3D plot of results) or axis object (to enable 3D plot of results).
    :param plot_2d_axis: "off" (default, to disable 2D plot of results) or axis object (to enable 2D plot of results).

    Output
    :return: ordered intersection curve, same format as int_curves ([body, element, x, y, z]).
    """

    # Preallocate output: matrices of intersection curves for each body
    int_curve_ordered = {}

    # Get array of unique element numbers in int_curve
    bodies = np.unique(int_curves[:, 0])

    # Iterate through all bodies
    for b in bodies:

        # Initialise the ordered/organised intersection curve for the current body
        body_int_curve_ordered = np.empty([0, 5])

        # Get intersection curves for just the current body
        body_indices = [i for i, j in enumerate(int_curves[:, 0]) if j == b]
        body_int_curve = int_curves[body_indices, :]

        # Remove the ignored elements
        select_elem_indices = [i for i, j in enumerate(body_int_curve[:, 1]) if int(j) not in ignore_elems]
        body_int_curve = body_int_curve[select_elem_indices, :]

        # List of all remaining elements
        body_elems = np.unique(body_int_curve[:, 1])

        # Piece together the curves in order
        # Start with the first point of the first element (move on to next element if first one is ignored)
        e = 0
        elem = body_elems[e]
        for i in range(len(body_elems)):
            if elem in ignore_elems:
                e += 1
                elem = body_elems[e]

        # Get the coordinates of the [second endpoint] of the curve (?)
        first_pt_indices = [j for j, k in enumerate(body_int_curve[:, 1]) if k == body_elems[e]]  # if bodies match
        end_pt_2_coords = body_int_curve[first_pt_indices[-1], 2:]  # CHANGED FROM [0]

        # Iterate through the number of (not-ignored) elements in the body to connect all curve segments
        for i in range(len(body_elems)):
            elem = body_elems[e]

            # Get the end points of the intersection curve
            body_elem_indices = [j for j, k in enumerate(body_int_curve[:, 1]) if k == elem]

            # Test new "end_pt_1" against old "end_pt_2"
            if i == 0:
                end_pt_1_coords = body_int_curve[body_elem_indices[0], 2:]
                end_pt_2_coords = body_int_curve[body_elem_indices[-1], 2:]
            else:
                dist1 = np.linalg.norm(body_int_curve[body_elem_indices[0], 2:] - end_pt_2_coords)
                dist2 = np.linalg.norm(body_int_curve[body_elem_indices[-1], 2:] - end_pt_2_coords)
                #if np.allclose(end_pt_2_coords, body_int_curve[body_elem_indices[0], 2:], atol=1e-2):
                if dist2 > dist1:
                    end_pt_1_coords = body_int_curve[body_elem_indices[0], 2:]
                    end_pt_2_coords = body_int_curve[body_elem_indices[-1], 2:]
                else:
                    # Flip the curve direction
                    end_pt_1_coords = body_int_curve[body_elem_indices[-1], 2:]
                    end_pt_2_coords = body_int_curve[body_elem_indices[0], 2:]
                    body_elem_indices = np.flip(body_elem_indices)

            # print("\n")
            # print(i, elem)
            # print("end_pt_1_coords:", end_pt_1_coords)
            # print("end_pt_2_coords:", end_pt_2_coords)

            # Store the ordered data
            body_int_curve_ordered = np.append(body_int_curve_ordered,
                                               body_int_curve[body_elem_indices[0:-1], :], axis=0)

            # Find matching coords and update e
            mindist = 1e6
            for j in range(body_int_curve.shape[0]):
                #if np.allclose(body_int_curve[j, 2:], end_pt_2_coords, atol=1e-2):
                dist = np.linalg.norm(body_int_curve[j, 2:] - end_pt_2_coords)
                if dist < mindist:
                    match_elem = body_int_curve[j, 1]
                    # print("elem, match_elem:", elem, match_elem, match_elem == elem)
                    if not match_elem == elem:
                        mindist = dist
                        # print("elem, match_elem:", elem, match_elem, "ACCEPTED")
                        e = np.where(body_elems == match_elem)

        # Add the last point to make complete curve
        body_int_curve_ordered = np.append(body_int_curve_ordered, [body_int_curve[body_elem_indices[-1], :]], axis=0)

        # Add body curve to the output
        int_curve_ordered[int(b)] = body_int_curve_ordered

        # Plotting the complete intersection curve
        for k in int_curve_ordered.keys():
            if not plot_3d_axis == "off":
                plot_3d_axis.plot3D(int_curve_ordered[k][:, 2], int_curve_ordered[k][:, 3], int_curve_ordered[k][:, 4],
                                    color="black")
            if not plot_2d_axis == "off":
                plot_2d_axis.plot(int_curve_ordered[b][:, 2], int_curve_ordered[b][:, 3], color="red")

                # add points at element boundaries
                cum_counts = np.empty(len(int_curve_ordered[b][:, 1]))
                for i in range(len(int_curve_ordered[b][:, 1])):
                    cum_counts[i] = np.sum(int_curve_ordered[b][:i, 1] == int_curve_ordered[b][i, 1])
                first_indices = cum_counts == 0
                plot_2d_axis.plot(int_curve_ordered[b][first_indices, 2], int_curve_ordered[b][first_indices, 3],
                                  marker=".", ms=1, lw=0, color="black")

    return int_curve_ordered


def get_intersection_curves(e_data, a_data, n_data, plane, nICpts, ignore_elements, plot_3d=False, plot_2d=False):
    """Generates the intersection curve of a bicubic Hermite element with a generic plane.

    Input
    :param e_data: element data for all bodies, format [body, element, node1.version, ..., node4.version].
    :param a_data: apex node data for all elements. 0 for quadrilateral, 1 or 3 for triangle.
    :param n_data: node data for all bodies, format [node.version, x, dx/ds1, dx/ds2, d2x/ds1ds2,
                                                                   y, dy/ds1, dy/ds2, d2y/ds1ds2,
                                                                   z, dz/ds1, dz/ds2, d2z/ds1ds2].
    :param: plane: intersection plane coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0).
    :param ignore_elements: list of element numbers to ignore from the intersection curve.
    :param plot_3d: boolean to enable/disable 3D plot of results.
    :param plot_2d: boolean to enable/disable 2D plot of results.

    Output
    :return: ordered intersection curves for each body, format [body, element, x, y, z].
    """

    # Initialise the figures (if turned on)
    [ax3, ax2] = initialise_figures(plot_3d, plot_2d)

    # Initialise the intersection curve and retrieve the curve for each element
    int_curves = np.empty([0, 5])
    for e in range(len(e_data)):  #range(2): #

        # Retrieve the body and element numbers for the index
        body = e_data[e, 0]
        elem = e_data[e, 1]

        # Get the nodal variables for the element
        elem_node_data = get_element_node_data(e, e_data, n_data)

        # Generate the intersection curve
        # print("Element: ", elem)
        segments = elemcurve.get_intersection_curve(elem_node_data, a_data[e], plane, nICpts, [body, elem])

        # Plot the element and curves
        if not ax3 == "off":
            [ex, ey, ez] = chq.get_whole_element(10, elem_node_data)
            ax3.plot_surface(ex, ey, ez, linewidth=0, antialiased=False, alpha=0.25)
            for s in range(len(segments)):
                ax3.plot3D(segments[s][:, 0], segments[s][:, 1], segments[s][:, 2], color="black", linestyle='dashed')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.set_xlim(-25, 300)
            ax3.set_ylim(50, 250)
            ax3.set_zlim(-325, 25)
            set_axes_equal(ax3)

        if not ax2 == "off":
            for s in range(len(segments)):
                ax2.plot(segments[s][:, 0], segments[s][:, 1], color="red", linestyle='dashed')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_aspect('equal')

        # Test if the curve segment contains any points and add those points to the intersection curve
        for seg, segment in enumerate(segments):
            if segment.size > 0:
                # elem_segment = np.concatenate((np.ones([segment.shape[0], 1])*[body, elem], segment), axis=1)
                # elem_segment = np.concatenate((np.ones([segment.shape[0], 1])*[body, elem, seg], segment), axis=1)
                elem_segment = np.concatenate((np.ones([segment.shape[0], 1])*[body, elem+seg/10.], segment), axis=1)
                int_curves = np.append(int_curves, elem_segment, axis=0)

    # with np.printoptions(threshold=np.inf):
    #     print("Unordered int curves:\n", int_curves)

    # Organise all of the element intersection curves
    int_curves_ordered = get_organised_intersection_curves(int_curves, ignore_elements, plot_3d_axis=ax3, plot_2d_axis=ax2)

    if plot_3d or plot_2d:
        plt.show()

    return int_curves_ordered


def get_element_node_data(e, e_data, n_data):
    """Retrieves the nodal data for each node in an element.

    Input
    :param e: row index of the element in e_data.
    :param e_data: element data for all bodies, format [body, element, node1.version, ..., node4.version].
    :param n_data: node data for all bodies, format [node.version, x, dx/ds1, dx/ds2, d2x/ds1ds2,
                                                                   y, dy/ds1, dy/ds2, d2y/ds1ds2,
                                                                   z, dz/ds1, dz/ds2, d2z/ds1ds2].

    Output
    :return: nodal data for the element, format [node1_data, ..., node2_data] where nodeX_data is a column of
             [x, dx/ds1, dx/ds2, d2x/ds1ds2, y, dy/ds1, dy/ds2, d2y/ds1ds2, z, dz/ds1, dz/ds2, d2z/ds1ds2] at node X.
    """

    # Preallocate output (3 directions x 4 variables) x (4 nodes)
    elem_node_data = np.zeros([12, 4])

    # Retrieve x-direction variables
    for n in range(4):
        node_index = [i for i, j in enumerate(n_data[:, 0]) if j == round(e_data[e, n + 2], 1)][0]
        elem_node_data[0:3, n] = n_data[node_index, 1:4].T

    # Retrieve y-direction variables
    for n in range(4):  # for y-vals
        node_index = [i for i, j in enumerate(n_data[:, 0]) if j == round(e_data[e, n + 6], 1)][0]
        elem_node_data[4:7, n] = n_data[node_index, 5:8].T

    # Retrieve z-direction variables
    for n in range(4):  # for z-vals
        node_index = [i for i, j in enumerate(n_data[:, 0]) if j == round(e_data[e, n + 10], 1)][0]
        elem_node_data[8:11, n] = n_data[node_index, 9:12].T

    elem_node_data = [elem_node_data[:, 0], elem_node_data[:, 1], elem_node_data[:, 2], elem_node_data[:, 3]]

    return elem_node_data


def initialise_figures(show_3d_plot, show_2d_plot):
    """Initialises the 3D and/or 2D figures to plot the intersection curves results.

    Input
    :param show_3d_plot: boolean to enable/disable the 3D data plot.
    :param show_2d_plot: boolean to enable/disable the 2D data plot.

    Output
    :return: list of axes objects [ax3, ax2] for the 3D and 2D data plots, respectively.
    """

    # Initialise the figures
    if show_3d_plot:
        fig1 = plt.figure(figsize=(1000/96, 800/96), dpi=96)
        axis_3 = plt.axes(projection='3d')
    else:
        axis_3 = "off"

    if show_2d_plot:
        fig2 = plt.figure(figsize=(1000/96, 800/96), dpi=96)
        axis_2 = plt.axes()
    else:
        axis_2 = "off"

    return [axis_3, axis_2]

def convert_intcurves_to_plane_local(global_coords, plane):
    """Transforms global 3D coordinates (int_curves format) into local 2D coordinates of a plane.
       see: https://stackoverflow.com/questions/49769459/convert-points-on-a-3d-plane-to-2d-coordinates

    Input
    :param global_coords: global coordinates for each body, format [body][x, y, z].
    :param plane: plane: list of coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0).

    Output
    :return: local_coords: numpy array of coordinates for each body. Returns None if conversion error greater than 1E-6.
    """
    #
    # see: https://stackoverflow.com/questions/49769459/convert-points-on-a-3d-plane-to-2d-coordinates

    # Unpack plane coeffcients
    [A, B, C, D] = plane

    # Vectors and unit vectors of plane coordinate system
    z_prime_v = np.array([A, B, C])
    z_prime_uv = z_prime_v/np.sqrt(np.sum(z_prime_v**2))
    if B != 0:
        y_prime_v = np.array([0, C/B, 1.])
    else:
        y_prime_v = np.array([0, 1., 0])
    y_prime_uv = y_prime_v/np.sqrt(np.sum(y_prime_v**2))
    x_prime_v = np.cross(y_prime_uv, z_prime_uv)
    x_prime_uv = x_prime_v/np.sqrt(np.sum(x_prime_v ** 2))

    # Basis points
    # origin-prime is the closest point on the plane to the origin
    o_prime_bp = -D/(A**2 + B**2 + C**2)*np.array([A, B, C])
    x_prime_bp = o_prime_bp + x_prime_uv
    y_prime_bp = o_prime_bp + y_prime_uv
    z_prime_bp = o_prime_bp + z_prime_uv

    # Get transform matrix (M) by solving at basis points
    S = np.concatenate((np.array([x_prime_bp, y_prime_bp, z_prime_bp, o_prime_bp]).T, np.ones([1, 4])))
    Dmat = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [1., 1., 1., 1.]])
    M = np.matmul(Dmat, np.linalg.inv(S))

    # Do transform
    local_coords = {}  # pre-allocate
    max_oop = 0 # out-of-plane sum to check results
    for b in range(len(global_coords)):
        local_coords[b] = np.empty([np.shape(global_coords[b])[0], 4])  # pre-allocate
        for p in range(np.shape(global_coords[b])[0]):
            affine_vector = np.transpose([np.concatenate((global_coords[b][p, :], [1.]))])  # vector for affine transform
            local_coords[b][p, :] = np.matmul(M, affine_vector).T[0]  # apply transform
        max_oop += max(max_oop, np.max(np.absolute(local_coords[b][:, 2])))

    # # Plotting and validation...
    # fig = plt.figure()
    # ax = plt.axes(projection='3d', proj_type = 'ortho')
    # ax.plot(int_curves[0][:, 2], int_curves[0][:, 3], int_curves[0][:, 4])
    # ax.plot(int_curves[1][:, 2], int_curves[1][:, 3], int_curves[1][:, 4])
    # assmcurve.set_axes_equal(ax)
    # ax.view_init(elev=69.5, azim=90)
    # # ax.view_init(elev=0, azim=0)
    #
    # # Check the result manually for plane = [0, 0.5, 1, 10]
    # theta = np.absolute(np.arctan(plane[1]/plane[2]))
    # print(theta)
    # o_dist = plane[3]*np.cos(theta)
    # check_coords_y = -o_dist*np.sin(theta) + local_coords[0][:, 1]*np.cos(theta) + local_coords[0][:, 2]*np.sin(theta)
    # check_coords_z = -o_dist*np.cos(theta) + local_coords[0][:, 1]*np.sin(theta) + local_coords[0][:, 2]*np.cos(theta)
    # diff_y = check_coords_y - int_curves[0][:, 3]
    # diff_z = check_coords_z - int_curves[0][:, 4]
    # print(np.min(diff_y), np.max(diff_y))
    # print(np.min(diff_z), np.max(diff_z))

    tol = 1E-6
    if max_oop < tol:
        local_coords = [local_coords[b][:, :2] for b in range(len(local_coords))]
        return local_coords
    else:
        print("Error in conversion to local coordinates: max out-of-plane value greater than tolerance.")
        return []