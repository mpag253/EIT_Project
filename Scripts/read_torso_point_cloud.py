import math
import numpy as np
import matplotlib.pyplot as plt

# # Read the mesh template .ipelem file
# file_ipe = open('templatetorsoNEWIDS.ipelem', 'r')
# lines_ipe = file_ipe.readlines()
# e_data = np.empty((0, 5))
# for ln, line in enumerate(lines_ipe):
#     if line.startswith(" Element number"):
#         elem_id = int(line.split()[-1])
#         temp1 = lines_ipe[ln+5].split()[-4:]
#         temp = np.concatenate(([elem_id], temp1))
#         e_data = np.append(e_data, [temp], axis=0)
# file_ipe.close()

# def get_ellipse_dydx(x, a, b):
#     dydx = -(b**2 * x) / (a**2 * math.sqrt((b**2 * (1-x**2/a**2))))
#     return dydx


def get_ellipse_y(x, a, b):
    y = b*math.sqrt(1 - (x/a)**2)
    return y


def get_lung_node_z(file, node_num):
    file_n = open(file, 'r')
    lines_n = file_n.readlines()
    for ln, line in enumerate(lines_n):
        if line.startswith("   z.  Value"):
            splitline = line.split()
            if splitline[-2] == "#Versions=":
                nvers = int(line.split()[-1])
            else:
                nvers = 1
        if line.startswith(" Node:"):
            if int(line.split()[-1]) == node_num:
                zpos = [2*math.ceil(nvers*4/5.) + math.ceil((nvers*4-3)/5.), (nvers*4-3)%5-1]
                zval = float(lines_n[ln+zpos[0]].split()[zpos[1]])
    file_n.close()
    return zval


# Read the mesh template .exnode file
file_exn = open('templatetorsoNEWIDS.exnode', 'r')
lines_exn = file_exn.readlines()
n_data = np.empty((0, 13))
for ln, line in enumerate(lines_exn):
    if line.startswith(" Node:"):
        node_id = int(line.split()[-1])
        temp1 = lines_exn[ln+1].split()
        temp2 = lines_exn[ln+2].split()
        temp3 = lines_exn[ln+3].split()
        temp = np.concatenate(([node_id], temp1, temp2, temp3))
        n_data = np.append(n_data, [temp], axis=0)
file_exn.close()
n_data = n_data[n_data[:, 0].astype(int).argsort()]

# Get landmarks from the corresponding fitted lung mesh
lung_path = "C:/Users/mipag_000/Documents/EIT_Project/TorsoFitting/Human_Aging/AGING003/EIsupine/Lung/"
lung_rn = lung_path + "Right_fitted.exnode"
minmax_nodes = [56, 96, 7, 49]  # [min_left, max_left, min_right, max_right]
lung_r_zmin = get_lung_node_z(lung_rn, minmax_nodes[2])
lung_r_zmax = get_lung_node_z(lung_rn, minmax_nodes[3])

# Retrieve torso data and evaluate the z-bounds
torso_data = np.loadtxt("TorsoFitting/surface_Torsotrimmed.ipdata", skiprows=1)
z_min = np.min(torso_data[:, 3])
z_max = np.max(torso_data[:, 3])
z_rng = z_max - z_min

# Evaluate scaling parameters for the mesh
tol = 20.0
mask = (torso_data[:, 3] < (z_min + tol))
bot_ring = torso_data[mask, :]
bot_min_x = np.min(bot_ring[:, 1])
bot_max_x = np.max(bot_ring[:, 1])
bot_min_y = np.min(bot_ring[:, 2])
bot_max_y = np.max(bot_ring[:, 2])
centre = [(bot_min_x + bot_max_x)/2, (bot_min_y + bot_max_y)/2]
boxdim = [(bot_max_x - bot_min_x)/2, (bot_max_y - bot_min_y)/2]  # [a, b] for ellipse

# Define node groups for constraints
nrings = 7
nod_rings = np.empty((nrings,16))
for i in range(nrings):
    nod_rings[i, :] = np.arange(i*16+1, i*16+17)
nod_cols = np.empty((16, nrings))
for i in range(16):
    nod_cols[i, :] = np.arange(i+1, 113, 16)

# Define x/a, y-dir for each node column
nod_col_vals = np.empty((16, 2))
nod_col_vals[0, :] = [-0.95, -1]
nod_col_vals[1, :] = [-0.65, -1]
nod_col_vals[2, :] = [-0.35, -1]
nod_col_vals[3, :] = [+0.00, -1]
nod_col_vals[4, :] = [+0.35, -1]
nod_col_vals[5, :] = [+0.65, -1]
nod_col_vals[6, :] = [+0.95, -1]
nod_col_vals[7, :] = [+0.95, +1]
nod_col_vals[8, :] = [+0.70, +1]
nod_col_vals[9, :] = [+0.45, +1]
nod_col_vals[10, :] = [+0.20, +1]
nod_col_vals[11, :] = [+0.00, +1]
nod_col_vals[12, :] = [-0.20, +1]
nod_col_vals[13, :] = [-0.45, +1]
nod_col_vals[14, :] = [-0.70, +1]
nod_col_vals[15, :] = [-0.95, +1]

# Calculate and store new node coordinates
for c in range(np.shape(nod_cols)[0]):
    nx = centre[0] + nod_col_vals[c, 0]*boxdim[0]
    ny = centre[1] + get_ellipse_y(nx-centre[0], boxdim[0], boxdim[1])*nod_col_vals[c, 1]
    # ndydx = get_ellipse_dydx(nx-centre[0], boxdim[0], boxdim[1])
    for i, n in enumerate(nod_cols[c, :].astype(int)):
        n_data[n-1, 1:] = 0.
        n_data[n-1, 1] = nx
        n_data[n-1, 5] = ny
z_buffer = 25.
for r in range(nrings):
    for i, n in enumerate(nod_rings[r, :].astype(int)):
        if r == nrings-1:
            n_data[n-1, 9] = lung_r_zmin - z_buffer
        elif r == 0:
            n_data[n - 1, 9] = z_max + z_buffer
        elif r == nrings-2:
            n_data[n - 1, 9] = lung_r_zmin + z_buffer
        elif r == 1:
            n_data[n - 1, 9] = lung_r_zmax
        else:
            n_data[n - 1, 9] = lung_r_zmax - (lung_r_zmax - lung_r_zmin - z_buffer)/(nrings-3)*(r-1)
if z_min < np.min(n_data[:, 9].astype(float)):
    print('WARNING: Data exists below mesh. Consider larger z_buffer or cropping data.')
if z_max > np.max(n_data[:, 9].astype(float)):
    print('WARNING: Data exists above mesh. Consider larger z_buffer or cropping data.')

# Write to file
for ln, line in enumerate(lines_exn):
    if line.startswith(" Node:"):
        node_id = int(line.split()[-1])
        new_data = n_data[node_id-97, 1:].astype(float)
        lines_exn[ln+1] = "  {:>13.6f}{:>13.6f}{:>13.6f}{:>13.6f}\n".format(new_data[0], new_data[1], new_data[2], new_data[3])
        lines_exn[ln+2] = "  {:>13.6f}{:>13.6f}{:>13.6f}{:>13.6f}\n".format(new_data[4], new_data[5], new_data[6], new_data[7])
        lines_exn[ln+3] = "  {:>13.6f}{:>13.6f}{:>13.6f}{:>13.6f}\n".format(new_data[8], new_data[9], new_data[10], new_data[11])
file_out = open('templatetorsoTIDIED.exnode', 'w')
file_out.writelines(lines_exn)
file_out.close()


plt.figure()
plt.scatter(bot_ring[:, 1], bot_ring[:, 2])
plt.plot([bot_min_x, bot_min_x, bot_max_x, bot_max_x, bot_min_x],
         [bot_min_y, bot_max_y, bot_max_y, bot_min_y, bot_min_y], 'r:')
plt.scatter(centre[0], centre[1], marker='x', c='r')
plt.scatter(n_data[:, 1].astype(float), n_data[:, 5].astype(float), marker='o', c='g')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(n_data[:, 1].astype(float), n_data[:, 5].astype(float), n_data[:, 9].astype(float), marker='o', c='g')
plt.show()



