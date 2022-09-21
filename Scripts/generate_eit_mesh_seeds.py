import BSplineParameterisation.bspline_parameterisation as bsparam
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
from sys import exit
from shapely.geometry import Polygon


path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E'

# version = 'truth'
# version = 'sample_mean'
version = 'predicted'

# Reference area to scale torso (float or None to not scale)
ref_areas = {'A':7.7044e4, 'B':7.6991e4, 'C':7.8618e4, 'D':7.2842e4, 'E':5.9548e4}
if version == 'truth':
    ref_area = None
else:
    ref_area = ref_areas[pca_id.split('-')[-1]]
    print('\nReference area: ', ref_area)

# Shape rotation
if version == 'truth':
    rotate = None  # clockwise rotation [radians]
else:
    rotate = -0.0729  # clockwise rotation [radians]

# Whether to plot the generated seeds
plot_seeds = False

# Define the subject IDs for the Leave-Out-Out
subject_ids = {'A':'H5977', 'B':'AGING043', 'C':'H7395', 'D':'AGING014', 'E':'AGING053'}

# B-spline data for mean shape
if version == 'sample_mean':
    subdir = 'sample_mean'
    mean_bspline_data = np.load('Geom\\pca_' + pca_id + '\\'+subdir+'\\sample_bsplines_mean.npy')
    [Is_body, Js_body, Fs_mean, Ks_mean] = mean_bspline_data.T
    seed_lungs = False
    seed_size = 2 #2      # seed size for the mesh (i.e. node spacing)
elif version == 'truth':
    subject_id = subject_ids[pca_id.split('-')[-1]]
    subdir = 'truth_'+subject_id
    truth_bspline_data = np.load('Geom\\pca_' + pca_id + '\\'+subdir+'\\truth_bsplines.npy')
    [Is_body, Js_body, Fs_mean, Ks_mean] = truth_bspline_data.T
    seed_lungs = True
    seed_size = 2 #1.5,6 inv:2      # seed size for the mesh (i.e. node spacing)
elif version == 'predicted':
    subject_id = subject_ids[pca_id.split('-')[-1]]
    subdir = 'predicted_' + subject_id
    predicted_bspline_data = np.load('Geom\\pca_' + pca_id + '\\'+subdir+'\\predicted_bsplines_mean.npy')
    [Is_body, Js_body, Fs_mean, Ks_mean] = predicted_bspline_data.T
    seed_lungs = False
    seed_size = 2  #2      # seed size for the mesh (i.e. node spacing)


def generate_preliminary_seeds(Fs, knots, phis):

    rphi = bsparam.evaluate_mesh_bspline(Fs, knots, phis)
    preseed_x = np.reshape(rphi*-np.cos(phis), [len(phis), 1])
    preseed_y = np.reshape(rphi*-np.sin(phis), [len(phis), 1])
    preseed = np.concatenate((preseed_x, preseed_y), axis=1)
    # print(torso_mean_polygon, np.shape(torso_mean_polygon))

    # Option to plot
    # plt.figure()
    # plt.plot(body_coords[:, 0], body_coords[:, 1])
    # plt.plot(torso_mean_polygon[:, 0], torso_mean_polygon[:, 1])
    # plt.show()

    # Option to scale area to reference mesh
    if ref_area is not None:
        p_area = Polygon(preseed).area
        print('\nInitial area: ', p_area)
        scale_factor = np.sqrt(ref_area/p_area)
    else:
        scale_factor = 1.

    return preseed, scale_factor


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
    resamp_rphi = bsparam.evaluate_mesh_bspline(Fs, knots, resamp_phis)
    # print("torso_mean_rphi:\n", torso_mean_rphi)
    resamp_seeds_x = np.reshape(resamp_rphi*-np.cos(resamp_phis), [len(resamp_phis), 1])
    resamp_seeds_y = np.reshape(resamp_rphi*-np.sin(resamp_phis), [len(resamp_phis), 1])
    resamp_seeds = np.hstack((resamp_seeds_x, resamp_seeds_y))
    # print("resamp_seeds:\n", resamp_seeds, np.shape(resamp_seeds))

    return resamp_seeds


def generate_mesh_seeds(Fs, knots, seed_size):

    # Generate preliminary mesh seeds
    phis = np.linspace(0, 2*pi, 1000)
    preseed, scale_factor = generate_preliminary_seeds(Fs, knots, phis)
    preseed *= scale_factor

    # Resample phi at desired seed size
    reseed = generate_resampled_seeds(Fs, knots, phis, preseed, seed_size)
    reseed *= scale_factor

    if ref_area is not None:
        p_area = Polygon(reseed).area
        print('\nScaled area: ', p_area)

    # Rotate the mesh
    if rotate is not None:
        rotmat = [[np.cos(rotate),-np.sin(rotate)],[np.sin(rotate),np.cos(rotate)]]
        reseed = np.matmul(reseed,rotmat)

    return reseed, scale_factor


def generate_mesh_seeds_lung(Fss, knots, seed_size, scale_factor):

    Fs_x = Fss[0]
    Fs_y = Fss[1]

    # Generate preliminary mesh seeds
    phis = np.linspace(0, 1, 1000)
    preseed_x = bsparam.evaluate_mesh_bspline(Fs_x, knots, phis)
    preseed_y = bsparam.evaluate_mesh_bspline(Fs_y, knots, phis)
    # print("preseed:\n", preseed_x, np.shape(preseed_x))
    preseed = np.hstack((np.reshape(preseed_x, [len(phis), 1]),
                         np.reshape(preseed_y, [len(phis), 1])))
    preseed *= scale_factor
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
    reseed *= scale_factor
    # print("reseed:\n", reseed, np.shape(reseed))

    # Rotate the mesh
    if rotate is not None:
        rotmat = [[np.cos(rotate),-np.sin(rotate)],[np.sin(rotate),np.cos(rotate)]]
        reseed = np.matmul(reseed,rotmat)

    return reseed


def export_seeds(path, file_id, seeds):
    filename = path+'mesh_seeds_'+file_id+'_{:04d}.csv'.format(int(seed_size*10))
    np.savetxt(filename, seeds, delimiter=",")
    print('\n', '='*80, '\n\n',
          '{:^80s}'.format('MESH SEEDS GENERATED'), '\n',
          '{:^80s}'.format('"'+filename+'"'), '\n\n',
          '{:^80s}'.format('NOW GO TO MATLAB TO GENERATE MESH:'), '\n',
          '{:^80s}'.format('"generate_polygon_trimesh.m"'), '\n\n',
          '='*80, '\n')
    return


def main():

    # Torso b-spline
    Fs_t = Fs_mean[Is_body==2]
    Ks_t = Ks_mean[Is_body==2]
    seeds_t, scale_factor = generate_mesh_seeds(Fs_t, Ks_t, seed_size)
    export_dir = 'Geom\\pca_' + pca_id + '\\' + subdir + '\\mesh_seeds\\'
    os.makedirs(export_dir, exist_ok=True)
    export_seeds(export_dir, 'torso', seeds_t)

    # Lungs
    # Right Lung
    Fs_lr_x = Fs_mean[(Is_body == 0)*(Js_body == 1)]
    Fs_lr_y = Fs_mean[(Is_body == 0)*(Js_body == 2)]
    Ks_lr = Ks_mean[(Is_body == 0)*(Js_body == 1)]
    # Left Lung
    Fs_ll_x = Fs_mean[(Is_body == 1)*(Js_body == 1)]
    Fs_ll_y = Fs_mean[(Is_body == 1)*(Js_body == 2)]
    Ks_ll = Ks_mean[(Is_body == 1)*(Js_body == 1)]
    if seed_lungs:
        # Right Lung
        seeds_lr = generate_mesh_seeds_lung([Fs_lr_x, Fs_lr_y], Ks_lr, seed_size, scale_factor)
        export_seeds(export_dir, 'lung_right', seeds_lr)
        # Left Lung
        seeds_ll = generate_mesh_seeds_lung([Fs_ll_x, Fs_ll_y], Ks_ll, seed_size, scale_factor)
        export_seeds(export_dir, 'lung_left', seeds_ll)

    # Plot
    if plot_seeds:
        phis_t = np.linspace(0, 2*np.pi, 200)
        eval_t_r = bsparam.evaluate_mesh_bspline(Fs_t, Ks_t, phis_t)
        eval_t_x = eval_t_r*-np.cos(phis_t)
        eval_t_y = eval_t_r*-np.sin(phis_t)
        phis_l = np.linspace(0, 1, 200)
        eval_lr_x = bsparam.evaluate_mesh_bspline(Fs_lr_x, Ks_lr, phis_l)
        eval_lr_y = bsparam.evaluate_mesh_bspline(Fs_lr_y, Ks_lr, phis_l)
        eval_ll_x = bsparam.evaluate_mesh_bspline(Fs_ll_x, Ks_lr, phis_l)
        eval_ll_y = bsparam.evaluate_mesh_bspline(Fs_ll_y, Ks_lr, phis_l)
        # Scale and rotate
        if rotate is not None:
            rotmat = [[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]]
        else:
            rotmat = [[1, 0], [0, 1]]
        eval_t_sr = scale_factor*np.matmul(np.vstack((eval_t_x, eval_t_y)).T, rotmat)
        eval_lr_sr = scale_factor*np.matmul(np.vstack((eval_lr_x, eval_lr_y)).T, rotmat)
        eval_ll_sr = scale_factor*np.matmul(np.vstack((eval_ll_x, eval_ll_y)).T, rotmat)
        # Figure
        plt.figure()
        # plt.plot(eval_t_x, eval_t_y, c='grey')
        plt.fill(eval_t_sr[:, 0], eval_t_sr[:, 1], facecolor='whitesmoke')
        plt.plot(eval_t_sr[:,0], eval_t_sr[:,1], c='black')
        # plt.plot(seeds_t[:, 0], seeds_t[:, 1], lw=0, marker='.')
        # plt.plot(eval_lr_x, eval_lr_y, c='grey')
        # plt.plot(eval_ll_x, eval_ll_y, c='grey')
        # plt.fill(eval_lr_sr[:, 0], eval_lr_sr[:, 1], facecolor='lightgrey')
        # plt.fill(eval_ll_sr[:, 0], eval_ll_sr[:, 1], facecolor='lightgrey')
        plt.plot(eval_lr_sr[:,0], eval_lr_sr[:,1], c='black')
        plt.plot(eval_ll_sr[:,0], eval_ll_sr[:,1], c='black')
        # if seed_lungs:
        #     plt.plot(seeds_lr[:, 0], seeds_lr[:, 1], lw=0, marker='.')
        #     plt.plot(seeds_ll[:, 0], seeds_ll[:, 1], lw=0, marker='.')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-200, 200)
        plt.ylim(-150, 150)
        plt.title(version)
        plt.show()


if __name__ == "__main__":
    main()