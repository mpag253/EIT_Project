import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import copy

import Slice2D.read_in_data as read_in_data

# INPUTS ###############################################################################################################

path = "C:\\Users\\mipag_000\\OneDrive - The University of Auckland\\Documents\\EIT_Project\\"

# pca_id = 'LRT_S_Mfull_N81_R-AGING001-EIsupine'
pca_id = 'LRT_S_Mfull_N80_R-AGING001-EIsupine_LOO-E'

# Get the subject ID for the Leave-Out-Out
subject_ids = {'A':'H5977', 'B':'AGING043', 'C':'H7395', 'D':'AGING014', 'E':'AGING053'}
subject_id = subject_ids[pca_id.split('-')[-1]]

# Load in elements
# elem_filenames = ["Geom\\template\\templateRight.exelem",
#                   "Geom\\template\\templateLeft.exelem",
#                   "Geom\\template\\templateTorso.exelem"]
# [eData, _, aData] = read_in_data.from_exelem(path, elem_filenames)

# Predicted Nodes
# nfiles_pred = ["Geom\\pca_" + pca_id + "\\sample_mean\\geom\\realigned\\fittedRight_realigned.exnode",
#                "Geom\\pca_" + pca_id + "\\sample_mean\\geom\\realigned\\fittedLeft_realigned.exnode",
#                "Geom\\pca_" + pca_id + "\\sample_mean\\geom\\realigned\\fittedTorso_realigned.exnode"]
nfiles_pred = ["Geom\\pca_" + pca_id + "\\predicted_" + subject_id + "\\geom\\realigned\\fittedRight_realigned.exnode",
               "Geom\\pca_" + pca_id + "\\predicted_" + subject_id + "\\geom\\realigned\\fittedLeft_realigned.exnode",
               "Geom\\pca_" + pca_id + "\\predicted_" + subject_id + "\\geom\\realigned\\fittedTorso_realigned.exnode"]
nData_pred = read_in_data.from_exnode(path, nfiles_pred)

# Predicted nodal covariance
Gamma_wgivenp = np.load("Geom\\pca_" + pca_id + "\\pca_condcov_wgivenp_"+pca_id+".npy")
modeshapes = np.load("Geom\\pca_" + pca_id + "\\pca_modeshapes_"+pca_id+".npy")
Gamma_mgivenp = np.diag(np.ones(np.shape(modeshapes)[1]))
Gamma_mgivenp[:5,:5] = Gamma_wgivenp
Gamma_ngivenp = np.matmul(np.matmul(modeshapes,Gamma_mgivenp),modeshapes.T)

# # print(eData)
# print(nData_pred)
# print(np.shape(nData_pred[:,1:].flatten()))
# print(modeshapes)
# # for i in np.sum(modeshapes,axis=1):
# #     print(i)
# print(np.shape(Gamma_mgivenp))
# print(np.shape(modeshapes))
# print(np.shape(Gamma_ngivenp))

# Truth nodes
nfiles_true = ["Geom\\pca_" + pca_id + "\\truth_" + subject_id + "\\geom\\Right_fitted_tf.exnode",
               "Geom\\pca_" + pca_id + "\\truth_" + subject_id + "\\geom\\Left_fitted_tf.exnode",
               "Geom\\pca_" + pca_id + "\\truth_" + subject_id + "\\geom\\Torso_fitted_tf.exnode"]
nData_true = read_in_data.from_exnode(path, nfiles_true)

def do_rotate_ndata_x(nData_, angle_x):
    rotmat = np.array([[np.cos(angle_x), -np.sin(angle_x)],
                       [np.sin(angle_x),  np.cos(angle_x)]])
    nData_[:, [y_idx, z_idx]] = np.matmul(nData_[:, [y_idx, z_idx]], rotmat)
    return nData_

def do_shift_ndata_x(nData_, shift_x):
    nData_[:, x_idx] += shift_x
    return nData_

def do_shift_ndata_y(nData_, shift_y):
    nData_[:, y_idx] += shift_y
    return nData_

def do_shift_ndata_z(nData_, shift_z):
    nData_[:, z_idx] += shift_z
    return nData_

def do_scale_ndata(nData_, scale):
    nData_[:, 1:] *= scale
    return nData_


# Shift to origin
top_ant_idx = np.where(nData_pred[:,0]==196.1)[0][0]
top_pos_idx = np.where(nData_pred[:,0]==204.1)[0][0]
top_lft_idx = np.where(nData_pred[:,0]==200.1)[0][0]
top_rgt_idx = np.where(nData_pred[:,0]==208.1)[0][0]
bot_ant_idx = np.where(nData_pred[:,0]==100.1)[0][0]
bot_pos_idx = np.where(nData_pred[:,0]==108.1)[0][0]
bot_lft_idx = np.where(nData_pred[:,0]==104.1)[0][0]
bot_rgt_idx = np.where(nData_pred[:,0]==112.1)[0][0]
x_idx = 1
y_idx = 5
z_idx = 9

# rotate truth about x
cc_dy_true = nData_true[top_lft_idx,y_idx]-nData_true[bot_lft_idx,y_idx]
cc_dz_true = nData_true[top_lft_idx,z_idx]-nData_true[bot_lft_idx,z_idx]
cc_angle_true = np.arctan(cc_dy_true/cc_dz_true)
cc_dy_pred = nData_pred[top_lft_idx,y_idx]-nData_pred[bot_lft_idx,y_idx]
cc_dz_pred = nData_pred[top_lft_idx,z_idx]-nData_pred[bot_lft_idx,z_idx]
cc_angle_pred = np.arctan(cc_dy_pred/cc_dz_pred)
rotx_angle = cc_angle_pred - cc_angle_true
print(rotx_angle)
nData_true = do_rotate_ndata_x(nData_true, rotx_angle)

# shift predicted to origin
shift_x_pred = -nData_pred[bot_ant_idx,x_idx]
shift_y_pred = -nData_pred[bot_lft_idx,y_idx]
shift_z_pred = -nData_pred[top_ant_idx,z_idx]
print(shift_x_pred, shift_y_pred, shift_z_pred)
# nData_pred[:,x_idx] += shift_x_pred
# nData_pred[:,y_idx] += shift_y_pred
# nData_pred[:,z_idx] += shift_z_pred
nData_pred = do_shift_ndata_x(nData_pred, shift_x_pred)
nData_pred = do_shift_ndata_y(nData_pred, shift_y_pred)
nData_pred = do_shift_ndata_z(nData_pred, shift_z_pred)
# shift truth to origin
shift_x_true = -nData_true[bot_ant_idx,x_idx]
shift_y_true = -nData_true[bot_lft_idx,y_idx]
shift_z_true = -nData_true[top_ant_idx,z_idx]
# nData_true[:,x_idx] += shift_x_true
# nData_true[:,y_idx] += shift_y_true
# nData_true[:,z_idx] += shift_z_true
nData_true = do_shift_ndata_x(nData_true, shift_x_true)
nData_true = do_shift_ndata_y(nData_true, shift_y_true)
nData_true = do_shift_ndata_z(nData_true, shift_z_true)

print(shift_x_true, shift_y_true, shift_z_true)

# Scale
ht_pred = nData_pred[top_ant_idx,z_idx] - nData_pred[bot_ant_idx,z_idx]
ht_true = nData_true[top_ant_idx,z_idx] - nData_true[bot_ant_idx,z_idx]
scale_t2p = ht_pred/ht_true
print(scale_t2p)
# scale true to the prediction
# nData_true[:, 1:] *= scale_t2p
nData_true = do_scale_ndata(nData_true, scale_t2p)
print(nData_pred[top_ant_idx,z_idx])
print(nData_pred[bot_ant_idx,z_idx])
print(nData_true[top_ant_idx,z_idx])
print(nData_true[bot_ant_idx,z_idx])

print(nData_pred[top_ant_idx,:])
print(nData_true[top_ant_idx,:])

# # Remove second derivs
# # nData_pred = nData_pred[:,[1,2,3,5,6,7,9,10,11]].flatten()
# # nData_true = nData_true[:,[1,2,3,5,6,7,9,10,11]].flatten()
# # del_indices = list(range(3, Gamma_ngivenp.shape[0], 4))
# # print(np.where(np.diag(Gamma_ngivenp)==0)[0])
# print(np.sqrt(np.diag(Gamma_ngivenp)))
# del_indices = np.where(np.diag(Gamma_ngivenp)==0)[0]
# Gamma_ngivenp = np.delete(Gamma_ngivenp, del_indices, axis=0)
# Gamma_ngivenp = np.delete(Gamma_ngivenp, del_indices, axis=1)
# diff_pred = nData_true[:,1:].flatten() - nData_pred[:,1:].flatten()
# diff_pred = np.delete(diff_pred, del_indices)
# print(diff_pred)

def get_keep_dofs_indices(ndof):
    keep_indices = np.empty([0])
    for dof in [0,4,8]:
        keep_indices = np.concatenate((keep_indices, np.arange(dof, ndof, 12)))
    keep_dofs_indices = np.isin(np.arange(ndof),keep_indices)
    return keep_dofs_indices

def delete_dofs(diff,Gamma,keep_dofs_indices,keep_nvers_indices):
    keep_indices = np.logical_and(keep_dofs_indices,keep_nvers_indices)
    del_indices = np.logical_not(keep_indices)
    Gamma = np.delete(Gamma, del_indices, axis=0)
    Gamma = np.delete(Gamma, del_indices, axis=1)
    diff = np.delete(diff, del_indices)
    return diff, Gamma

def get_diff_nData(nData_1,nData_2):
    diff = (nData_1[:, 1:].flatten()) - (nData_2[:, 1:].flatten())
    return diff


# # Only keep coordinates
# # diff_pred = nData_true[:, 1:].flatten() - nData_pred[:, 1:].flatten()
# diff_pred = get_diff_nData(nData_true,nData_pred)
# # n_dof = np.shape(Gamma_ngivenp)[0]
# # print(n_dof)
# # # should be 197 nodes x3 --> 591 dof. VERSIONS!
# # del_indices = np.empty([0])
# # print(del_indices)
# # for dof in [1,2,3,5,6,7,9,10,11]:
# #     del_indices = np.concatenate((del_indices,np.arange(dof,n_dof,12)))
# # del_indices = del_indices.astype(int)
# # print(np.shape(del_indices))
# # Gamma_ngivenp = np.delete(Gamma_ngivenp, del_indices, axis=0)
# # Gamma_ngivenp = np.delete(Gamma_ngivenp, del_indices, axis=1)
# # diff_pred = np.delete(diff_pred, del_indices)
# # print(diff_pred)
# [diff_pred,Gamma_ngivenp] = delete_dofs(diff_pred,Gamma_ngivenp)

# print(np.linalg.cond(Gamma_ngivenp))

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(nData_true[:,x_idx],nData_true[:,y_idx],nData_true[:,z_idx],c='blue')
# ax.scatter(nData_pred[:,x_idx],nData_pred[:,y_idx],nData_pred[:,z_idx],c='red')
# plt.show()

x_indices = np.arange(0, 12, len(Gamma_ngivenp))
y_indices = np.arange(4, 12, len(Gamma_ngivenp))
z_indices = np.arange(8, 12, len(Gamma_ngivenp))

# stdevs = np.sqrt(np.diag(Gamma_ngivenp))
# print(np.shape(nData_true))
# plt.figure()
# plt.plot(nData_true[:,x_idx],c='blue')
# plt.plot(nData_pred[:,x_idx],c='red')
# plt.plot(nData_pred[:,x_idx]+stdevs[x_indices],c='red',ls='dashed')
# plt.plot(nData_pred[:,x_idx]-stdevs[x_indices],c='red',ls='dashed')
# # plt.figure()
# # plt.plot(nData_true[:,x_idx+1])
# # plt.plot(nData_pred[:,x_idx+1])
# # plt.figure()
# # plt.plot(nData_true[:,x_idx+2])
# # plt.plot(nData_pred[:,x_idx+2])
# plt.figure()
# plt.plot(nData_true[:,y_idx],c='blue')
# plt.plot(nData_pred[:,y_idx],c='red')
# plt.plot(nData_pred[:,y_idx]+stdevs[y_indices],c='red',ls='dashed')
# plt.plot(nData_pred[:,y_idx]-stdevs[y_indices],c='red',ls='dashed')
# # plt.figure()
# # plt.plot(nData_true[:,y_idx+1])
# # plt.plot(nData_pred[:,y_idx+1])
# # plt.figure()
# # plt.plot(nData_true[:,y_idx+2])
# # plt.plot(nData_pred[:,y_idx+2])
# plt.show()

# GUESSTIMATE OF M-D
# print(np.sum(np.abs(np.divide(diff_pred,np.sqrt(np.diag(Gamma_ngivenp))))))
# plt.figure()
# plt.plot(np.divide(diff_pred,np.sqrt(np.diag(Gamma_ngivenp))))
# plt.show()

# print(np.min(diff_pred))
# print(np.max(diff_pred))
# print(np.min(np.sqrt(np.diag(Gamma_ngivenp))))
# print(np.max(np.sqrt(np.diag(Gamma_ngivenp))))

def get_md(diff,Gamma):
    # # Gamma_inv = np.linalg.inv(Gamma + 1e-12*np.diag(np.ones(len(Gamma))))
    # # md = np.sqrt(np.matmul(diff.T, np.matmul(Gamma_inv, diff)))
    # Gamma += 1e-12*np.diag(np.ones(len(Gamma)))
    md = np.sqrt(np.matmul(diff.T, np.linalg.solve(Gamma, diff)))
    print(md)
    return md

def get_re(diff):
    re = np.linalg.norm(diff)
    print(re)
    return re

def get_rmse(diff):
    n = int(len(diff)/3)
    dists = np.empty((n))
    for i in range(n):
        dists[i] = np.linalg.norm(diff[3*i:3*i+3])
    rmse = np.sqrt(np.mean(np.square(dists)))
    print(rmse)
    return rmse

# # MD
# diff_pred = np.reshape(diff_pred,[len(diff_pred),1])
# print(np.shape(diff_pred))
# print(np.shape(Gamma_ngivenp))
# md_pred = get_md(diff_pred,Gamma_ngivenp)
# print(md_pred)

# def min_md_func(shift_y):
#     diff_pred_new = diff_pred
#     diff_pred_new[y_indices] += shift_y
#     md_pred_new = get_md(diff_pred_new, Gamma_ngivenp)
#     return md_pred_new

# def min_re_func(shift_y):
#     diff_pred_new = diff_pred
#     diff_pred_new[y_indices] += shift_y
#     re_pred_new = get_re(diff_pred_new)
#     return re_pred_new

# shift_ys = range(-20,20,1)
# # shift_mds = np.empty(np.shape(shift_ys))
# shift_res = np.empty(np.shape(shift_ys))
# for i in range(len(shift_ys)):
#     # shift_mds[i] = min_md_func(shift_ys[i])
#     shift_res[i] = min_re_func(shift_ys[i])
# plt.figure()
# # plt.plot(shift_ys,shift_mds)
# plt.plot(shift_ys,shift_res)
# plt.show()

def prepare_data(nData_true_new, nData_pred, Gamma_ngivenp):
    diff_pred = get_diff_nData(nData_true_new,nData_pred)
    keep_nvers_indices = get_keep_nvers_indices(nData_true_new)
    keep_dofs_indices = get_keep_dofs_indices(len(Gamma_ngivenp))
    # print(np.sum(keep_nvers_indices))
    # print(np.sum(keep_dofs_indices))
    [diff_pred_keep,Gamma_ngivenp_keep] = delete_dofs(diff_pred,Gamma_ngivenp,keep_dofs_indices,keep_nvers_indices)
    # print(np.shape(diff_pred_keep))
    # print(np.shape(Gamma_ngivenp_keep))
    return diff_pred_keep, Gamma_ngivenp_keep

def get_keep_nvers_indices(nData_):
    nvers = nData_[:,0].reshape([len(nData_[:,0]),1])
    nvers_list = np.matmul(nvers,np.ones((1,12))).flatten()
    nvers_floor = np.floor(nvers_list)
    nvers_floor_unqiue = np.unique(nvers_floor)
    keep_nvers = np.empty(len(nvers_floor_unqiue))
    for i, nver in enumerate(nvers_floor_unqiue):
        keep_nvers[i] = max(nvers_list[nvers_floor==nver])
    keep_nvers_indices = np.isin(nvers_list,keep_nvers)
    return keep_nvers_indices

def do_transforms(nData_true,x):
    nData_true_new = copy.deepcopy(nData_true)
    nData_true_new = do_rotate_ndata_x(nData_true_new, x[0])
    nData_true_new = do_shift_ndata_y(nData_true_new, x[1])
    nData_true_new = do_shift_ndata_z(nData_true_new, x[2])
    nData_true_new = do_scale_ndata(nData_true_new, x[3])
    return nData_true_new

def min_func(x,nData_true,nData_pred,Gamma_ngivenp):
    # [angle_x,shift_y,shift_z,scale] = x
    # nData_true_new = copy.deepcopy(nData_true)
    nData_true_new = do_transforms(nData_true,x)
    # diff_pred_new = get_diff_nData(nData_true_new,nData_pred)
    # [diff_pred_new,Gamma_ngivenp_keep] = delete_dofs(diff_pred_new,Gamma_ngivenp)
    [diff_pred_keep, Gamma_ngivenp_keep] = prepare_data(nData_true_new, nData_pred, Gamma_ngivenp)
    # metric = get_re(diff_pred_keep)
    # metric = get_md(diff_pred_keep,Gamma_ngivenp_keep)
    metric = get_rmse(diff_pred_keep)
    return metric

# plt.figure()
# plt.plot(diff_pred)
# plt.show()

# Nelder-Mead, Powell, BFGS
res = spo.minimize(min_func,
                   [0.0,0.0,0.0,1.0],
                   args=(nData_true,nData_pred,Gamma_ngivenp),
                   method='Nelder-Mead',
                   options={'maxiter':100000})
# print(res)
print("\nSuccess?", res.success)
print("RMSE:", res.fun)
print("X:", res.x)
resx = res.x

# print(np.shape(nData_true))
# print(np.shape(nData_true.flatten()))
# print(np.shape(nData_true[:,1:].flatten()))
# print(np.shape(Gamma_ngivenp))


# # Manual
# # start:
# print('Start: ',min_re_func([0,0,0,0],nData_true,nData_pred,Gamma_ngivenp))
# nevals = 11
# angle_zs = np.linspace(-0.1,0.1,num=nevals)
# shift_ys = np.linspace(-50,50,num=nevals)
# shift_zs = np.linspace(-50,50,num=nevals)
# scales = np.linspace(0.9,1.1,num=nevals)
# evals = np.empty([nevals,nevals,nevals,nevals])
# for i, x0 in enumerate(angle_zs):
#     for j, x1 in enumerate(shift_ys):
#         for k, x2 in enumerate(shift_zs):
#             for l, x3 in enumerate(scales):
#                 x = [x0,x1,x2,x3]
#                 # print(nData_true[0,5],nData_pred[0,5])
#                 evals[i,j,k,l] = min_func(x,nData_true,nData_pred,Gamma_ngivenp)
# opt = evals.min()
# print(opt)
# oi = np.where(evals==opt)
# print(oi)
# resx = [angle_zs[oi[0][0]],
#         shift_ys[oi[1][0]],
#         shift_zs[oi[2][0]],
#         scales[oi[3][0]] ]
# print(resx)


nData_true_opt = do_transforms(nData_true,resx)
# print(nData_true_opt)
# print(np.shape(nData_true_opt))

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(nData_true_opt[:,x_idx],nData_true_opt[:,y_idx],nData_true_opt[:,z_idx],c='blue')
ax.scatter(nData_pred[:,x_idx],nData_pred[:,y_idx],nData_pred[:,z_idx],c='red')
plt.show()

# relative error better
# m-d still bad
# get rmse
