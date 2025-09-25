import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
import colorsys
from skimage import color

def plot_colormap_trajectory(colormap_name, color_space='RGB'):
    """
    Plots the 3D color space trajectory of a Matplotlib colormap.

    Args:
        colormap_name (str): The name of the Matplotlib colormap to use.
        color_space (str): The color space to plot ('RGB', 'HSV', or 'LAB').
    """
    if color_space.upper() not in ['RGB', 'HSV', 'LAB']:
        print(f"Error: '{color_space}' is not a supported color space. Please choose 'RGB', 'HSV', or 'LAB'.")
        return

    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        print(f"Error: '{colormap_name}' is not a valid colormap name.")
        return

    x = np.linspace(0, 1, 256)
    rgb_colors_norm = cmap(x)[:, :3]

    if color_space.upper() == 'RGB':
        plot_colors = rgb_colors_norm
        axis_labels = ['Red', 'Green', 'Blue']
    elif color_space.upper() == 'HSV':
        plot_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in rgb_colors_norm])
        axis_labels = ['Hue', 'Saturation', 'Value']
    elif color_space.upper() == 'LAB':
        plot_colors = color.rgb2lab(rgb_colors_norm)
        axis_labels = ['L*', 'a*', 'b*']
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(plot_colors[:, 0], plot_colors[:, 1], plot_colors[:, 2], color='gray', linewidth=1)
    
    ax.scatter(plot_colors[:, 0], plot_colors[:, 1], plot_colors[:, 2], c=rgb_colors_norm, marker='o', s=20)
    
    start_point = plot_colors[0]
    end_point = plot_colors[-1]
    ax.scatter(start_point[0], start_point[1], start_point[2], c='green', marker='^', s=100, label='Start')
    ax.scatter(end_point[0], end_point[1], end_point[2], c='red', marker='v', s=100, label='End')
    ax.text(start_point[0], start_point[1], start_point[2], '  Start', color='black')
    ax.text(end_point[0], end_point[1], end_point[2], '  End', color='black')
    
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(f"'{colormap_name}' Colormap Trajectory in '{color_space}' Space")

    cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])
    
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('Normalized Value (0.0 to 1.0)')
    
    # --- NEW CODE: Remove ticks from the colorbar ---
    cbar.set_ticks([])
    # --- END OF NEW CODE ---
    
    plt.show()

if __name__ == '__main__':
    plot_colormap_trajectory('viridis')
    plot_colormap_trajectory('plasma')
    plot_colormap_trajectory('hsv')
    plot_colormap_trajectory('bwr')
    plot_colormap_trajectory('twilight')
    plot_colormap_trajectory('jet')