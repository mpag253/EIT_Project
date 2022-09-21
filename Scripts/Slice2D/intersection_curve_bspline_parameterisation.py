import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from . import element_plane_intersection_curve as elemcurve
from . import assembly_plane_intersection_curve as assmcurve
from . import read_in_data
import scipy.interpolate as ip


def print_in_full(thing):
    with np.printoptions(threshold=np.inf):
        print(thing)


def bspline_basis_1(phi, knots):
    return ((phi - knots[1])/(knots[3] - knots[1]))*((phi - knots[1])/(knots[2] - knots[1]))


def bspline_basis_2(phi, knots):
    return (((phi - knots[0])/(knots[2] - knots[0]))*((knots[2] - phi)/(knots[2] - knots[1])) +
            ((knots[3] - phi)/(knots[3] - knots[1]))*((phi - knots[1])/(knots[2] - knots[1])))


def bspline_basis_3(phi, knots):
    return ((knots[2] - phi)/(knots[2] - knots[0]))*((knots[2] - phi)/(knots[2] - knots[1]))


def plot_psi_i(knots, ax):
    num_pts = 10
    phis_A = np.linspace(knots[0], knots[1], num_pts)
    phis_B = np.linspace(knots[1], knots[2], num_pts)
    phis_C = np.linspace(knots[2], knots[3], num_pts)

    psi_i_A = np.empty([len(phis_A)])
    for j in range(len(phis_A)):
        phi = phis_A[j]
        psi_i_A[j] = bspline_basis_1(phi, knots)

    psi_i_B = np.empty([len(phis_B)])
    for j in range(len(phis_B)):
        phi = phis_B[j]
        psi_i_B[j] = bspline_basis_2(phi, knots)

    psi_i_C = np.empty([len(phis_C)])
    for j in range(len(phis_C)):
        phi = phis_C[j]
        psi_i_C[j] = bspline_basis_3(phi, knots)

    ax.plot(np.concatenate((phis_A, phis_B, phis_C)), np.concatenate((psi_i_A, psi_i_B, psi_i_C)))

    # return [np.concatenate((phis_A, phis_B, phis_C)), np.concatenate((psi_i_A, psi_i_B, psi_i_C))]
    return


def get_local_knots(phi, mesh):

    js = [p for p, q in enumerate(mesh) if q < phi]

    if not js:
        j = -1
        local_knots = np.concatenate((mesh[-2:] - 2*np.pi, mesh[0:2]))

    else:
        j = js[-1]

        if j == 0:
            local_knots = np.concatenate(([mesh[-1] - 2*np.pi], mesh[0:3]))

        elif j == (len(mesh) - 1):
            local_knots = np.concatenate((mesh[-2:], mesh[0:2] + 2*np.pi))

        elif j == (len(mesh) - 2):
            local_knots = np.concatenate((mesh[-3:], [mesh[0] + 2*np.pi]))

        else:
            local_knots = mesh[j - 1:j + 3]

    return [local_knots, j]


def get_polar_coordinates(cartesian_coords):

    polar_r = np.sqrt(np.square(cartesian_coords[:, 0]) + np.square(cartesian_coords[:, 1])).T
    polar_phi = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0]).T + np.pi
    polar_coords = np.concatenate(([polar_phi], [polar_r], [cartesian_coords[:, 2]])).T
    polar_coords = polar_coords[np.argsort(polar_coords[:, 0])]

    # Options to print truncated or in full
    # print(polar_coords)
    # print_in_full(polar_coords)

    return polar_coords


def convert_to_rho_space(polar_coordinates):

    rho_m = min(polar_coordinates[:, 1])
    rho_p = max(polar_coordinates[:, 1])
    rho_0 = (rho_m + rho_p)/2
    delta_rho = rho_p - rho_m

    polar_rho_coordinates = polar_coordinates
    polar_rho_coordinates[:, 1] = (polar_rho_coordinates[:, 1] - rho_0)/delta_rho

    return [polar_rho_coordinates, [rho_0, rho_m, rho_p, delta_rho]]


def get_fitted_spline_from_package(polar_coords, mesh, phis):

    # Add continuity between the two ends -- similar to argument "per=True"
    # added extra points to bounds to make the mesh knots work
    data_p = np.concatenate((polar_coords[-5:, 0] - 2*np.pi, polar_coords[:, 0], polar_coords[0:5, 0] + 2*np.pi))
    data_r = np.concatenate((polar_coords[-5:, 1], polar_coords[:, 1], polar_coords[0:5, 1]))
    # data_p2 = np.concatenate((polar_coords[:, 0], [polar_coords[0, 0] + 2*np.pi]))
    # data_r2 = np.concatenate((polar_coords[:, 1], [polar_coords[0, 1]]))

    # # Adding discontinuity --- NOT WORKING
    # # can do (1) by repeating the KNOT (1 repeat reduces order by 1) --- ERROR
    # # can also do by fitting two curves --- NOT WORKING: can't get "open" curve (curve to end at discont.)
    # discont_elem = 89
    # discont_data_idx = [i for i, j in enumerate(polar_coords[:, 2]) if j == discont_elem][-1]
    # discont_p = data_p[discont_data_idx]
    # discont_mesh_idx = [i for i, j in enumerate(mesh) if j > discont_p][0]
    # mesh = np.concatenate((mesh[:discont_mesh_idx], [discont_p, discont_p], mesh[discont_mesh_idx:]))

    spl = ip.splrep(data_p, data_r, k=3, t=mesh)
    # spl = ip.splrep(data_p, data_r, k=3)
    # spl2 = ip.splrep(data_p2, data_r2, k=3, per=True)

    r_spline = ip.splev(phis, spl)
    # r_spline2 = ip.splev(phis, spl2)

    return r_spline


def add_discontinuity_to_mesh(discontinuous_element, polar_coords, mesh):

    # Retrieve the index of the first node of the discontinuous element (i.e. the discontinuous node)
    discont_data_idx = [i for i, j in enumerate(polar_coords[:, 2]) if j == discontinuous_element][0]

    # Retrieve the coordinates of the discontinuous node
    discont_phi = polar_coords[discont_data_idx, 0]
    discont_rad = polar_coords[discont_data_idx, 1]

    # Retrieve the location to add the discontinuity (before)
    discont_mesh_idx = [i for i, j in enumerate(mesh) if j > discont_phi][0]

    # Add in two knots at the discontinuity
    mesh = np.concatenate((mesh[:discont_mesh_idx], [discont_phi, discont_phi], mesh[discont_mesh_idx:]))

    return mesh


def fit_mesh_bspline_to_data(data, mesh, discontinuous_elements=None, plot_bases=False):
    # determine the "weights" for a B-spline with knots defined by a mesh
    # the B-spline is fitted to the data using a linear least squares solution

    data_phi = data[:, 0]
    data_rad = data[:, 1]

    # Adding discontinuity: two repeated knots at the discontinuity (1 repeat reduces order by 1)
    if discontinuous_elements is not None:
        for discont_elem in discontinuous_elements:
            mesh = add_discontinuity_to_mesh(discont_elem, data, mesh)
        # print("\n\nLength of mesh:", len(mesh), "\n", mesh, "\n")

    if plot_bases:
        fig, ax = plt.subplots(subplot_kw={'projection': None})
        ax.set_ylim(-.1, 1.2)

    # Pre-allocate the coefficient matrix (A) for the linear least squares solution
    A = np.empty([len(data_phi), len(mesh)])
    # Pre-allocate array for the sum of the basis functions (for validation)
    basis_sum = np.zeros([len(data_phi)])

    # Iterate through all the points to fit and populate the coefficient matrix
    for i, phi in enumerate(data_phi):
        # Find the local knots that bound the current value of phi
        [local_knots, j] = get_local_knots(phi, mesh)
        # print(i, phi, j, len(mesh)-1, local_knots)

        # Update the coefficient matrix
        A[i, j + 0] = bspline_basis_1(phi, local_knots)
        A[i, j - 1] = bspline_basis_2(phi, local_knots)
        A[i, j - 2] = bspline_basis_3(phi, local_knots)

        # Plot the basis functions at this value of phi and evaluate the sum (for validation)
        if plot_bases:
            plot_psi_i(local_knots, ax)
        basis_sum[i] = A[i, j + 0] + A[i, j - 1] + A[i, j - 2]

    # PLot the sum of all basis functions (for validation)
    if plot_bases:
        ax.plot(data_phi, basis_sum, color='black', linestyle='dashed')

    # (Optional) Print the coefficient matrix (truncated or in full)
    print("Coefficient matrix, A:\n", A, "\n", np.shape(A), "\n", )
    # print_in_full("Coefficient matrix, A:\n", A, "\n", np.shape(A), "\n", )

    # Solve the linear least squares of the b-spline to the data
    Fs = np.linalg.lstsq(A, np.transpose([data_rad]), rcond=None)

    return [Fs, mesh]


def evaluate_mesh_bspline(mesh, Fs, phis):
    # reproduce the splines for a given array of phi (phis)

    # Pre-allocate an array for the soloution
    rphi = np.empty([len(phis)])

    # Iterate through each value of phi
    for i, phi in enumerate(phis):
        # Find the local knots that bound the current value of phi
        [local_knots, j] = get_local_knots(phi, mesh)
        # print(i, phi, j, local_knots)

        # Evaluate the value of the spline at phi using the basis functions and fitted weights
        rphi[i] = (Fs[j - 0]*bspline_basis_1(phi, local_knots) +
                   Fs[j - 1]*bspline_basis_2(phi, local_knots) +
                   Fs[j - 2]*bspline_basis_3(phi, local_knots))

    return rphi


def plot_splines_rectilinear_axes(xs, ys, legend):

    fig, ax = plt.subplots(subplot_kw={'projection': None})
    ax.fill_between(xs[0], ys[0], color=(0.9, 0.9, 0.9), label='_nolegend_')
    ax.plot([0, 2*np.pi], [-.5, -.5], color="red", linestyle='dashed')
    ax.plot([0, 2*np.pi], [0.5, 0.5], color="red", linestyle='dashed', label='_nolegend_')
    ax.plot([0, 2*np.pi], [0.0, 0.0], color="red", linestyle='dashed', label='_nolegend_')

    colours = ["black", "lime", "orange", "purple", "black", "black", "black"]
    lines = ["solid", "dashed", "dashed", "dashed", "dashed", "dashed"]

    for i in range(len(ys)):
        ax.plot(xs[i], ys[i], color=colours[i], linestyle=lines[i])
    # ax4.plot(discont_phi, discont_rad, marker='o', color="red")

    ax.set_xlabel('Angle (\u03D5)')
    ax.set_ylabel('Radius (r)')
    # ax4.set_xlim(0.4, 1.2)
    # ax4.set_ylim(70, 100)
    plt.title('B-Spline fitting (polar coordinates)', fontsize=16, fontweight="bold")  # , y=1.12)
    ax.legend([r'$\rho_0, \rho_+, \rho_-$'] + legend, fontsize=14,
              ncol=1, frameon=True)  # , bbox_to_anchor=(0, 1, 1, 0.12), loc='upper center')


def plot_splines_polar_axes(xs, ys, rhos, legend):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.fill(xs[0], ys[0], c=(0.9, 0.9, 0.9), label='_nolegend_')
    ax.plot(np.linspace(0, 2*np.pi, 50), np.ones([50])*rhos[1], color="red", linestyle='dashed')
    ax.plot(np.linspace(0, 2*np.pi, 50), np.ones([50])*rhos[2], color="red", linestyle='dashed', label='_nolegend_')
    ax.plot(np.linspace(0, 2*np.pi, 50), np.ones([50])*rhos[0], color="red", linestyle='dashed', label='_nolegend_')

    colours = ["black", "lime", "orange", "purple", "black", "black", "black"]
    lines = ["solid", "dashed", "dashed", "dashed", "dashed", "dashed"]

    for i in range(len(ys)):
        ax.plot(xs[i], ys[i], color=colours[i], linestyle=lines[i])

    # ax.plot(polar_coords[:, 0], polar_coords[:, 1]*rho_params[3] + rho_params[0], color="black", linestyle='solid')
    # ax.plot(phis, r_spline*rho_params[3] + rho_params[0], color="lime", linestyle='dashed')
    # # ax.plot(phis, r_spline2*rho_params[3] + rho_params[0], color="green", linestyle='dashed')
    # ax.plot(phis, rphi*rho_params[3] + rho_params[0], color="orange", linestyle='dashed')
    # # ax.plot(discont_phi, discont_rad, marker='x', color="red")

    ax.set_xlabel('Angle (\u03D5)')
    ax.set_ylabel('Radius (r)')
    plt.title('B-Spline fitting (polar coordinates)', fontsize=14, fontweight="bold")  # , y=1.12)
    ax.legend([r'$\rho_0, \rho_+, \rho_-$', 'Raw shape',
               'Fited B-Spline'], fontsize=16,
              ncol=1, frameon=True)  # , bbox_to_anchor=(0, 1, 1, 0.12), loc='upper center')


def get_parameterised_curve(body_coords, mesh, discont_elems=None, plot_bsplines=False, plot_bases=False):

    # Convert curve into ordered polar coordinates for parameterisation and fitting
    polar_coords = get_polar_coordinates(body_coords)

    # Convert the radius component into rho-space
    [polar_coords, rho_params] = convert_to_rho_space(polar_coords)

    # Define an array of phi values for reproducing the spline (i.e. plotting)
    phis = np.linspace(0., 2.*np.pi, 1000)

    # # Using the scipy.interpolate B-spline functions (doesn't have full functionality)
    # r_spline = get_fitted_spline_from_package(polar_coords, mesh, phis)

    # Fit the splines (manually coded)
    # discont_elems = [85]
    [Fs, mesh] = fit_mesh_bspline_to_data(polar_coords, mesh, discont_elems, plot_bases=plot_bases)

    # Plot the results
    if plot_bsplines:

        # Evaluate the splines (manually coded)
        rphi = evaluate_mesh_bspline(mesh, Fs[0], phis)

        # Plotting polar coordinates on rectilinear axes
        xs = [polar_coords[:, 0], phis]
        ys = [polar_coords[:, 1], rphi]
        legend = ['Raw shape', 'Fitted B-Spline']
        plot_splines_rectilinear_axes(xs, ys, legend)

        # Plotting polar coordinates on polar axes
        xs = [polar_coords[:, 0], phis]
        ys = [polar_coords[:, 1]*rho_params[3] + rho_params[0], rphi*rho_params[3] + rho_params[0]]
        legend = ['Raw shape', 'Fitted B-Spline']
        plot_splines_polar_axes(xs, ys, rho_params, legend)

        plt.show()

    return [Fs, mesh]
