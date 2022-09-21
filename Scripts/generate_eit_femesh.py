import BSplineParameterisation.bspline_parameterisation as bsparam
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import parameterise_torso as parat
import parameterise_lung as paral
import pickle
from sys import exit


path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'

# Define the filenames of the ".exnode" and ".exelem" files for each body:

# --- POPULATION MEAN ---
nFile = 'TEST_population_sample_mean'
torso_node_fname = "Geom\\"+nFile+"\\fittedTorso.exnode"
torso_elem_fname = "Geom\\template\\templateTorso.exelem"
generate_mesh_seeds = True
seed_lungs = False
seed_size = 10      # seed size for the mesh (i.e. node spacing)

# # --- TRUTH ---
# nFile = 'TEST_HLA-H11303_truth'
# torso_node_fname = "Geom\\"+nFile+"\\Torso_fitted.exnode"
# torso_elem_fname = "Geom\\template\\templateTorso.exelem"
# lungl_node_fname = "Geom\\"+nFile+"\\Left_fitted.exnode"
# lungl_elem_fname = "Geom\\template\\templateLeft.exelem"
# lungr_node_fname = "Geom\\"+nFile+"\\Right_fitted.exnode"
# lungr_elem_fname = "Geom\\template\\templateRight.exelem"
# generate_mesh_seeds = True
# seed_lungs = True
# seed_size = 5      # seed size for the mesh (i.e. node spacing)

# # --- PREDICTED ---
# nFile = 'TEST_HLA-H11303_predicted'
# torso_node_fname = "Geom\\"+nFile+"\\fittedTorso.exnode"
# torso_elem_fname = "Geom\\template\\templateTorso.exelem"
# generate_mesh_seeds = False
# seed_lungs = False
# seed_size = 10      # seed size for the mesh (i.e. node spacing)


nknots = 50         # number of knots for B-spline fitting


def generate_preliminary_seeds(Fs, knots, phis):

    rphi = bsparam.evaluate_mesh_bspline(Fs[0], knots, phis)
    preseed_x = np.reshape(-rphi*np.cos(phis), [len(phis), 1])
    preseed_y = np.reshape(-rphi*np.sin(phis), [len(phis), 1])
    preseed = np.concatenate((preseed_x, preseed_y), axis=1)
    # print(torso_mean_polygon, np.shape(torso_mean_polygon))

    # Option to plot
    # plt.figure()
    # plt.plot(body_coords[:, 0], body_coords[:, 1])
    # plt.plot(torso_mean_polygon[:, 0], torso_mean_polygon[:, 1])
    # plt.show()

    return preseed


def generate_resampled_seeds(Fs, knots, phis, prelim_seeds, elem_size):

    perims = np.sqrt(np.sum(np.square(prelim_seeds[1:, :] - prelim_seeds[:-1, :]), axis=1))
    cumperim = np.hstack(([0], np.cumsum(perims)))
    # print(cumperim, np.shape(cumperim))
    n_perim_elems = int(np.ceil(cumperim[-1]/elem_size))
    resamp_cumperim = np.linspace(0, cumperim[-1], n_perim_elems + 1)[1:]
    # print("mesh cumperim:\n", mesh_cumperim, np.shape(mesh_cumperim))
    resamp_phis = np.interp(resamp_cumperim, cumperim, phis)
    # print("phis:\n", phis)
    # print("torso_mean_phis:\n", torso_mean_phis)
    resamp_rphi = bsparam.evaluate_mesh_bspline(Fs[0], knots, resamp_phis)
    # print("torso_mean_rphi:\n", torso_mean_rphi)
    resamp_seeds_x = np.reshape(-resamp_rphi*np.cos(resamp_phis), [len(resamp_phis), 1])
    resamp_seeds_y = np.reshape(-resamp_rphi*np.sin(resamp_phis), [len(resamp_phis), 1])
    resamp_seeds = np.hstack((resamp_seeds_x, resamp_seeds_y))
    # print("resamp_seeds:\n", resamp_seeds, np.shape(resamp_seeds))

    return resamp_seeds


def generate_mesh_seeds(Fs, knots, seed_size):

    # Generate preliminary mesh seeds
    phis = np.linspace(0, 2*pi, 100)
    preseed = generate_preliminary_seeds(Fs, knots, phis)

    # Resample phi at desired seed size
    reseed = generate_resampled_seeds(Fs, knots, phis, preseed, seed_size)

    return reseed


def generate_mesh_seeds_lung(params, seed_size):

    knots = params[:, 0]
    Fs_x = params[:, 1]
    Fs_y = params[:, 2]

    # Generate preliminary mesh seeds
    phis = np.linspace(0, 1, 100)
    preseed_x = bsparam.evaluate_mesh_bspline(Fs_x, knots, phis)
    preseed_y = bsparam.evaluate_mesh_bspline(Fs_y, knots, phis)
    # print("preseed:\n", preseed_x, np.shape(preseed_x))
    preseed = np.hstack((np.reshape(preseed_x, [len(phis), 1]),
                         np.reshape(preseed_y, [len(phis), 1])))
    # print("preseed:\n", preseed, np.shape(preseed))

    # Resample x and y at desired seed size
    perims = np.sqrt(np.sum(np.square(preseed[1:, :] - preseed[:-1, :]), axis=1))
    cumperim = np.hstack(([0], np.cumsum(perims)))
    # print(cumperim, np.shape(cumperim))
    n_perim_elems = int(np.ceil(cumperim[-1]/seed_size))
    resamp_cumperim = np.linspace(0, cumperim[-1], n_perim_elems + 1)[1:]
    # print("mesh cumperim:\n", mesh_cumperim, np.shape(mesh_cumperim))
    resamp_phis = np.interp(resamp_cumperim, cumperim, phis)
    # print("phis:\n", phis)
    # print("torso_mean_phis:\n", torso_mean_phis)
    reseed_x = bsparam.evaluate_mesh_bspline(Fs_x, knots, resamp_phis).T
    reseed_y = bsparam.evaluate_mesh_bspline(Fs_y, knots, resamp_phis).T
    reseed = np.hstack((np.reshape(reseed_x, [len(resamp_phis), 1]),
                        np.reshape(reseed_y, [len(resamp_phis), 1])))
    # print("reseed:\n", reseed, np.shape(reseed))

    return reseed


def export_seeds(path, file_id, seeds):
    filename = path+'meshseed_'+file_id+'_{:04d}.csv'.format(seed_size)
    np.savetxt(filename, seeds, delimiter=",")
    print('\n', '='*80, '\n\n',
          '{:^80s}'.format('MESH SEEDS GENERATED'), '\n',
          '{:^80s}'.format('"'+filename+'"'), '\n\n',
          '{:^80s}'.format('NOW GO TO MATLAB TO GENERATE MESH:'), '\n',
          '{:^80s}'.format('"Ru-EIT/Mitchell/generate_polygon_trimesh.m"'), '\n\n',
          '='*80, '\n')
    return


def main():

    # # Torso
    # file_id_t = 'torso'
    # [Fs_t, knots_t, centre_t, body_coords_t] = pt.parameterise_torso(path, torso_node_fname, torso_elem_fname, plane, nICpts, nknots, ignore_elems)
    # bsparam.export_bspline("Geom\\"+nFile+"\\", 'Torso', file_id_t, Fs_t, knots_t, shift=centre_t, rho_factor=1, base_name=nFile)
    # # Define the plane: coefficients [A, B, C, D] for plane equation (Ax + By + Cz + D = 0)
    # node_num_zmin = 100
    # node_num_zmax = 196
    # mesh_zmin = mean_nData[mean_nData[:, 0] == (node_num_zmin + 0.1), 9]
    # mesh_zmax = mean_nData[mean_nData[:, 0] == (node_num_zmax + 0.1), 9]
    # slice_z = -(mesh_zmin + 0.518*(mesh_zmax - mesh_zmin))
    # plane = [0., 0., 1., slice_z]

    file_id_t = 'torso'
    with open('Geom\\pca_'+pca_id+'\\mean_bsplines_data.pkl', 'rb') as f:
        bspline_data_mean = pickle.load(f)
    Fs_torso = bspline_data_mean['torso']['r']['Fs']
    Ks_torso = bspline_data_mean['torso']['r']['Ks']


    if generate_mesh_seeds:

        # Torso cont.
        seeds_t = generate_mesh_seeds(Fs_torso, ks_t, seed_size)
        export_seeds("Geom\\"+nFile+"\\", file_id_t, seeds_t)

        # Lungs
        if seed_lungs:

            # # Right Lung
            # file_id_r = 'lung_right'
            # # [params_r, body_coords_r] = paral.parameterise_lung(path, lungr_node_fname, lungr_elem_fname, plane, nICpts, nknots, ignore_elems, discont_elems[0])
            # seeds_r = generate_mesh_seeds_lung(params_r, seed_size)
            # body_coords_r[:, :2] -= centre_t
            # seeds_r -= centre_t
            # export_seeds("Geom\\" + nFile + "\\", file_id_r, seeds_r)
            Fs_lung_right_x = bspline_data_mean['torso']['x']['Fs']
            Fs_lung_right_y = bspline_data_mean['torso']['y']['Fs']
            Ks_lung_right = bspline_data_mean['torso']['x']['Ks']
            seeds_r = generate_mesh_seeds_lung(params_r, seed_size)


            # # Left Lung
            # file_id_l = 'lung_left'
            # # [params_l, body_coords_l] = paral.parameterise_lung(path, lungl_node_fname, lungl_elem_fname, plane, nICpts, nknots, ignore_elems, discont_elems[1])
            # seeds_l = generate_mesh_seeds_lung(params_l, seed_size)
            # body_coords_l[:, :2] -= centre_t
            # seeds_l -= centre_t
            # export_seeds("Geom\\"+nFile+"\\", file_id_l, seeds_l)
            Fs_lung_right_x = bspline_data_mean['torso']['x']['Fs']
            Fs_lung_right_y = bspline_data_mean['torso']['y']['Fs']
            Ks_lung_right = bspline_data_mean['torso']['x']['Ks']
            seeds_r = generate_mesh_seeds_lung(params_r, seed_size)



        # Plot
        plt.figure()
        plt.plot(body_coords_t[:, 0], body_coords_t[:, 1])
        plt.plot(seeds_t[:, 0], seeds_t[:, 1], lw=0, marker='.')
        if seed_lungs:
            plt.plot(body_coords_l[:, 0], body_coords_l[:, 1])
            plt.plot(seeds_l[:, 0], seeds_l[:, 1], lw=0, marker='.')
            plt.plot(body_coords_r[:, 0], body_coords_r[:, 1])
            plt.plot(seeds_r[:, 0], seeds_r[:, 1], lw=0, marker='.')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    # NOTE: need to install PyDistMesh
    # if generate_mesh:  # TEMPORARY - merge with generate_mesh_seeds

        # seeds_t
        # seeds_l
        # seeds_r
        # pv_t = seeds_t
        # pv_all = np.vstack((seeds_t, seeds_l, seeds_r))

        # MATLAB
        # bbox = 1.01*[min(pv_t(:, 1)) min(pv_t(:, 2)); max(pv_t(:, 1)) max(pv_t(:, 2))];
        # [nodes, tris] = distmesh2d(@dpoly, @huniform, mesh_size, bbox, pv_all, pv_t);
        # tgeom = triangulation(tris, nodes);

        # PYTHON
        # >> > import distmesh as dm
        # >> > fd = lambda p: dm.ddiff(dm.drectangle(p, -1, 1, -1, 1),
        #                              ...
        # dm.dcircle(p, 0, 0, 0.5))
        # >> > fh = lambda p: 0.05 + 0.3*dm.dcircle(p, 0, 0, 0.5)
        # >> > p, t = dm.distmesh2d(fd, fh, 0.05, (-1, -1, 1, 1),
        #                           ...[(-1, -1), (-1, 1), (1, -1), (1, 1)])

if __name__ == "__main__":
    main()