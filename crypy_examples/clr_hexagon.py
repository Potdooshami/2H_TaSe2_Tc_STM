import numpy as np
import matplotlib.pyplot as plt

#---------
def gen_chex(x=0,y=0,R=1, c_ord=['r','g','b'], thickness=[1,10], **kwargs):
    """
    Generates and plots a colored hexagon with alternating colors and thicknesses.

    Parameters:
    x (float): x coordinate of the center of the hexagon.
    y (float): y coordinate of the center of the hexagon.
    R (float): Radius of the hexagon.
    c_ord (list): List of colors to cycle through.
    thickness (list): List of line thicknesses to cycle through.
    **kwargs: Additional keyword arguments passed to plt.plot.
    """
    c_ord = np.array(c_ord)
    thickness = np.array(thickness)
    
    #---------
    # Generate angles for the 6 vertices of the hexagon (rotated by 30 degrees or pi/6)
    tht = (np.arange(6) + 0.5) * 2 * np.pi / 6
    
    # Calculate x and y coordinates of the vertices
    xs = np.cos(tht) * R +x
    ys = np.sin(tht) * R +y
    
    # Define indices for alternating thickness and cycling colors
    tind = [0, 1, 0, 1, 0, 1]
    cind = [0, 1, 2, 0, 1, 2]
    
    # Map the indices to the actual thickness and color arrays
    thickness6 = thickness[tind]
    c6 = c_ord[cind]
    
    # Loop through each side of the hexagon
    for ind, thk, clr in zip(range(6), thickness6, c6):
        # Select the current vertex and the next one (wrapping around using modulo)
        x2 = xs[[ind, (ind + 1) % 6]]
        y2 = ys[[ind, (ind + 1) % 6]]
        
        # Plot the line segment
        plt.plot(x2, y2, color=clr, linewidth=thk, **kwargs)

if __name__ == '__main__':
    gen_chex()
    plt.show()
    


