# This is a script to determine the shape mismatch (percent symmetric difference) between two polygons

# The script uses a function "get_shape_mismatch" to generate a plot of the geometries and the mismatch
# The inputs to the function are two array of x-y coordinates for the two geometries

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os


def get_shape_mismatch(geom_1, geom_2, geom_names=None, output_dir=None, show_plot=True, save_plot=False):
    # ...

    if geom_names is None:
        geom_names = ["Geometry_1", "Geometry_2"]

    # Initialise the plot
    fig = plt.figure()
    ax = plt.axes()
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.85])

    # Convert the input coordinates to shapely polygons
    geom_1_p = Polygon(geom_1)
    geom_2_p = Polygon(geom_2)

    # Generate geometries for the intersection and union of geom_1 and geom_2
    geom_inter = geom_1_p.intersection(geom_2_p)
    geom_union = geom_1_p.union(geom_2_p)

    # Calculate the shape mismatch
    geom_1_area = geom_1_p.area
    geom_sdiff_area = geom_union.area - geom_inter.area
    delta_s = geom_sdiff_area/geom_1_area
    # print('\nShape mismatch (%): ' + '{:.1f}'.format(delta_s*100) + '\n')

    # Alternate version and check (likely more efficient but not as useful for plotting)
    # geom_sdiff = geom_1_p.symmetric_difference(geom_2_p)
    # delta_s_check = geom_sdiff.area/geom_1_area

    # Generate the plot and show and/or save, as appropriate
    if show_plot or save_plot:

        [mdl_string_fwd, mdl_string_inv] = geom_names

        x, y = geom_union.exterior.xy
        ax.fill(x, y, c=(0.9, 0.9, 0.9), label='_nolegend_')

        x, y = geom_inter.exterior.xy
        ax.fill(x, y, c=(1.0, 1.0, 1.0), label='_nolegend_')

        x, y = geom_1_p.exterior.xy
        ax.plot(x, y, c=(0.0, 0.0, 0.0), linestyle='solid')

        x, y = geom_2_p.exterior.xy
        ax.plot(x, y, c=(0.0, 0.0, 0.0), linestyle='dashed')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Shape mismatch (\u0394S)', y=1.12, fontsize=18, fontweight="bold")

        ax.legend([mdl_string_fwd, mdl_string_inv],
                   ncol=2, frameon = False, bbox_to_anchor=(0, 1, 1, 0.12), loc='upper center')
        ax.set_aspect('equal', 'box')

        ax.text(0.5, 0.5, '\u0394S =\n' + '{:.2f}'.format(delta_s*100) + '%', fontsize=18,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if save_plot:
        output_filename = 'shape_mismatch_FWD_' + mdl_string_fwd + '_INV_' + mdl_string_inv + '.png'
        if output_dir is None:
            plt.savefig(output_filename)
        else:
            saveto = os.path.join(".\\Output", output_filename)
            plt.savefig(saveto)
    if show_plot:
        plt.show()
    else:
        plt.clf()

    return delta_s

