import numpy as np
import warnings
import math


def from_ipelem(filename):
    """[redundant - possible issues with versions]"""

    file_left = open(filename, 'r')
    lines_left = file_left.readlines()

    elem_count = -1
    for line in lines_left:

        # retrieve total number of elements
        # print(line[0:4])
        # print(line.startswith(" The number of elements is"))
        if line.startswith(" The number of elements is"):
            n_elems = int(line.split()[-1])
            # print("Number of elements: " + str(n_elems))
            element_data = np.zeros([n_elems, 5])

        # retrieve active element number
        if line.startswith(" Element number"):
            elem_count += 1
            elem_num = int(line.split()[-1])
            # print(elem_num)
            element_data[elem_count, 0] = elem_num

        if line.startswith(" Enter the 4 global numbers for basis 1:"):
            elem_nodes = list(map(int, line.split()[-4:]))
            elem_nvers = [float(i) for i in elem_nodes]
            elem_nvers = [x + 0.1 for x in elem_nvers]
            # print(elem_nodes)
            # print(elem_nvers)
            element_data[elem_count, 1:] = elem_nvers
            # print(element_data)

        if line.startswith(" The version number for occurrence"):
            version_data = line.split()
            # print(version_data)
            occurence_num = int(version_data[-9])
            occurence_nod = int(version_data[-6][0:-1])
            version_num = int(version_data[-1])
            njj_string = version_data[-5]
            # print(occurence_num, occurence_nod, version_num)

            if njj_string == "njj=1":
                # index for the nth occurence of an item in a list
                node_index = [i for i, n in enumerate(elem_nodes) if n == occurence_nod][occurence_num - 1]
                elem_nvers[node_index] += (version_num-1)*0.1
                element_data[elem_count, node_index + 1] += (version_num-1)/10.

        # previous_line = line
    return element_data


def from_ipnode(path, filenames):
    """[redundant - possible issues with versions]"""

    node_data = np.empty([0, 13])

    for filename in filenames:
        file = open(path+filename, 'r')
        lines = file.readlines()

        node_count = -1
        for line in lines:

            # retrieve total number of nodes
            if line.startswith(" The number of nodes is"):
                n_nodes = int(line.split()[-1])

            # retrieve active node number
            if line.startswith(" Node number"):
                node_count += 1
                node_num = int(line.split()[-1])
                ver_num = 1  # in case there is only one version

            if line.startswith(" The number of versions for nj=1"):
                n_versions = int(line.split()[-1])
                current_node_data = np.zeros([n_versions, 13])

            if line.startswith(" For version number"):
                ver_num = int(line.split()[-1][0:-1])

            if line.startswith(" The Xj("):
                coord_data = line.split()
                coord_dir = int(coord_data[1][3])
                coord_val = float(line.split()[-1])
                current_node_data[ver_num-1, 0] = node_num + ver_num/10.
                current_node_data[ver_num-1, 4*coord_dir - 3] = coord_val

            if line.startswith(" The derivative wrt direction 1"):

                line_list = line.split()
                if len(line_list) == 9:
                    deriv1_val = float(line_list[-1])
                else:
                    deriv1_val = 0.0
                    warnings.warn("Missing node value (derivative: 1, node: "+str(
                        node_num)+", file: "+filename+"). Setting value to 0.0.")
                current_node_data[(ver_num - 1), (4*coord_dir - 2)] = deriv1_val

            if line.startswith(" The derivative wrt direction 2"):
                line_list = line.split()
                if len(line_list) == 9:
                    deriv2_val = float(line.split()[-1])
                else:
                    deriv2_val = 0.0
                    warnings.warn("Missing node value (derivative: 2, node: " + str(
                        node_num) + ", file: " + filename + "). Setting value to 0.0.")
                current_node_data[ver_num - 1, 4*coord_dir - 1] = deriv2_val

            if line.startswith(" The derivative wrt directions 1 & 2"):
                line_list = line.split()
                if len(line_list) == 11:
                    deriv12_val = float(line.split()[-1])
                else:
                    deriv12_val = 0.0
                    warnings.warn("Missing node value (derivative: 1 & 2, node: " + str(
                        node_num) + ", file: " + filename + "). Setting value to 0.0.")
                current_node_data[ver_num - 1, 4*coord_dir] = deriv12_val

                # print(ver_num, n_versions, coord_dir)
                if (ver_num == n_versions) and (coord_dir == 3):
                    node_data = np.append(node_data, current_node_data, axis=0)

    return node_data


def from_exelem(path, filenames):
    """Reads in data from an '.exelem' file to a numpy array.

    Input
    :param path: path to the files.
    :param filenames: list of filenames to read as a single assembly.

    Output
    :return: numpy array of the node data, row format [body, element, node1.version, ..., node4.version].
    """

    element_data = np.empty([0, 14])
    scale_factors = np.empty([0, 17])

    for body, filename in enumerate(filenames):

        file = open(path + filename, 'r')
        lines = file.readlines()

        for i, line in enumerate(lines):  #141 or 201

            # retrieve the current number of nodes
            if line.startswith(" #Nodes="):
                n_global_nodes = int(line.split()[-1])
                versions = np.zeros([12])

            if line.startswith("     #Nodes="):
                n_local_nodes = int(line.split()[-1])

                # retrieve the current direction
                if lines[i-1].startswith("   x."):
                    current_dir = 1
                elif lines[i-1].startswith("   y."):
                    current_dir = 2
                elif lines[i-1].startswith("   z."):
                    current_dir = 3
                else:
                    print("ERROR: didnt get current coord direction!!!")

                # retrieve the current local node numbers
                local_node_nums = np.array([int(lines[i + 1].split()[0][0]), int(lines[i + 4].split()[0][0]),
                                            int(lines[i + 7].split()[0][0]), int(lines[i + 10].split()[0][0])])

                # retrieve the version number for current node and direction
                versions[4*(current_dir-1):4*(current_dir-1)+4] =\
                    np.array([int(lines[i + 2].split()[-1]), int(lines[i + 5].split()[-1]),
                              int(lines[i + 8].split()[-1]), int(lines[i + 11].split()[-1])])/4

            # retrieve the current element and all data
            if line.startswith(" Element:") and lines[i+1].startswith("   Faces:"):

                # retrieve the element number
                elem_num = int(line.split()[-3])

                # retrieve the global node numbers
                global_node_nums = np.array([int(j) for j in lines[i + 7].split()])

                # retrieve the scale factors for the current element
                current_scale_factors = [float(j) for j in lines[i + 9].split() + lines[i + 10].split() +
                                                           lines[i + 11].split() + lines[i + 12].split()]

                # organise and export all the data
                node_nums = [global_node_nums[j-1] for j in local_node_nums]
                node_num_vers = np.array(node_nums + node_nums + node_nums) + np.array(versions)/10.
                current_elem_data = np.insert(node_num_vers, 0, [body, elem_num], axis=0)
                element_data = np.append(element_data, [current_elem_data], axis=0)
                current_scale_factors = np.insert(current_scale_factors, 0, elem_num, axis=0)
                scale_factors = np.append(scale_factors, [current_scale_factors], axis=0)

    apex_nodes = np.zeros(len(element_data))
    for e in range(len(element_data)):
        repeat_n1 = np.any(element_data[e, 2] == element_data[e, 3:6])
        repeat_n3 = len(np.unique(np.floor(element_data[e, 3:6]))) < 3
        if repeat_n1:
            apex_nodes[e] = 1
        elif repeat_n3:
            apex_nodes[e] = 3
        else:
            apex_nodes[e] = 0

    return element_data, scale_factors, apex_nodes


def from_exnode(path, filenames):
    """Reads in data from an '.exnode' file to a numpy array.

    Input
    :param path: path to the files.
    :param filenames: list of filenames to read as a single assembly.

    Output
    :return: numpy array of the node data, row format [node.version, x, dx/ds1, dx/ds2, d2x/ds1ds2,
                                                                     y, dy/ds1, dy/ds2, d2y/ds1ds2,
                                                                     z, dz/ds1, dz/ds2, d2z/ds1ds2].
    """

    node_data = np.empty([0, 13])

    for filename in filenames:
        file = open(path+filename, 'r')
        lines = file.readlines()

        read_count = 0
        node_count = -1
        for line in lines:

            # retrieve numbers of versions for upcoming node
            if line.startswith("   x.") or line.startswith("  x."):
                split_line = line.split()
                if len(split_line) == 9:
                    n_vers = int(line.split()[-1])
                else:
                    n_vers = 1

            # retrieve the nodal data (check before retrieving node number)
            if 0 < read_count < n_read_lines+1:

                current_node_data = np.append(current_node_data, [float(i) for i in np.array(line.split())])

                # print(read_count, n_read_lines)
                if read_count == n_read_lines:
                    # print(current_node_data)
                    # print(node_num, n_vers, n_read_lines)
                    data_x = np.reshape(current_node_data[0:n_vers*4], (n_vers, 4))
                    data_y = np.reshape(current_node_data[n_vers*4:(n_vers+n_vers)*4], (n_vers, 4))
                    data_z = np.reshape(current_node_data[(n_vers + n_vers)*4:], (n_vers, 4))
                    current_node_data = np.concatenate((data_x, data_y, data_z), axis=1)
                    # print(current_node_data)

                    for i in range(n_vers):
                        node_ver = float(node_num) + 0.1*(i+1)
                        # print(current_node_data[i])
                        current_node_ver_data = np.insert(current_node_data[i], 0, node_ver, axis=0)
                        node_data = np.append(node_data, [current_node_ver_data], axis=0)

                    # n_vers = 1  # in case the next node doesn't specify

                read_count += 1

            # retrieve active node number
            if line.startswith(" Node:"):
                node_count += 1
                node_num = int(line.split()[-1])
                read_count = 1

                n_read_lines = 3*math.ceil(4.*n_vers/5.)
                # print(n_read_lines)

                current_node_data = np.empty([0])
                # print(current_node_data)

    return node_data

