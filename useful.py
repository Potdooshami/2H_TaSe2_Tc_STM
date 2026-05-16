import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.image
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def tickoff(ax = None):
    """
    Remove ticks from the current axes.
    """
    if ax is None:
        ax = plt.gca()
    
    ax.set_xticks([])
    ax.set_yticks([])

def gci():
    """
    Get the current image object.

    Returns
    -------
    matplotlib.image.AxesImage
        The most recently added image object on the current axes.
    """
    return plt.gca().images[-1]

def change_image_range(image_object, vmin=None, vmax=None):
    """
    Change the vmin and vmax of a matplotlib image object.

    Parameters
    ----------
    image_object : matplotlib.image.AxesImage
        The image object to modify.
    vmin : float, optional
        The new minimum value for color mapping. If None, current vmin is kept.
    vmax : float, optional
        The new maximum value for color mapping. If None, current vmax is kept.
    """
    if vmin is not None:
        image_object.set_clim(vmin=vmin)
    if vmax is not None:
        image_object.set_clim(vmax=vmax)

def auto_clim(data, method='percentile', lower=1, upper=99):    
    """
    Calculate color limits based on data distribution.

    Parameters
    ----------
    data : 2D array
        Input image data.
    method : str
        'percentile' or 'std'.
    lower/upper : float
        Percentile thresholds or sigma multiplier for std.
    """
    if method == 'percentile':
        vmin, vmax = np.percentile(data, [lower, upper])
    elif method == 'std':
        mean = np.mean(data)
        std = np.std(data)
        vmin, vmax = mean - std, mean + std
    else:
        raise ValueError("method must be 'percentile' or 'std'")
    return vmin, vmax

def auto199():
    """
    Automatically adjust the color limits of the current image to the 1% and 99% percentiles.
    """
    ic = gci()
    foo = ic.get_array()
    aclim = auto_clim(foo, method='percentile', lower=1, upper=99)
    change_image_range(ic, aclim[0], aclim[1])

def plot_result(image, background):
    """
    Plot the original image, background, and the result (image - background) side by side.
    """
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()
    return fig, ax

def subtabs(figs, titles=None):
    """
    Display multiple figures in an ipywidgets Tab environment.
    """
    import ipywidgets as widgets
    from IPython.display import display
    tabs = widgets.Tab()
    tabs.children = [widgets.Output() for _ in figs]
    for i, fig in enumerate(figs):
        with tabs.children[i]:
            display(fig)
    if titles is not None:
        for i, title in enumerate(titles):
            tabs.set_title(i, title)
    return tabs

def subtabFigSr(figsSr):
    """
    Generate tabs from a pandas Series containing Figure objects.
    """
    return subtabs(figsSr.to_list(), list(figsSr.index))

def subtabSr(arrSr):
    """
    Generate tabs from a pandas Series containing image arrays.
    
    Parameters
    ----------
    arrSr : pd.Series 
        Series containing numpy arrays to be plotted as images.
    """
    def figimshow(arr):        
        fig = plt.figure(figsize=(20,20))
        plt.imshow(arr, cmap='afmhot')
        try:
            auto199()
        except:
            pass
        plt.close(fig)        
        return fig
    
    figs = list(map(figimshow, list(arrSr.values)))
    titles = arrSr.index.to_list()
    tabs = subtabs(figs, titles)
    return tabs

def color_histogram(arr, clim=None, nbins=200, cmap='gray', ax=None):
    """
    Create and display a colored histogram for 1D array data.

    Args:
        arr (np.ndarray): Input 1D NumPy array.
        clim (tuple, optional): (min, max) tuple for colormap range. 
                                Defaults to array min/max.
        nbins (int, optional): Number of histogram bins. Defaults to 200.
        cmap (str or Colormap, optional): Matplotlib colormap to use. Defaults to 'gray'.
        ax (matplotlib.axes.Axes, optional): Target axes to plot on.
    """
    # Check if input data is 1D
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")

    # Set default clim
    if clim is None:
        clim = (arr.min(), arr.max())

    # Calculate histogram
    n, bins = np.histogram(arr, bins=nbins, range=clim)

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Get colormap
    colormap = get_cmap(cmap)

    # Normalize bin centers between 0 and 1
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    normalized_centers = norm(bin_centers)

    # Determine colors based on normalized values
    colors = colormap(normalized_centers)

    # Plot histogram
    if ax is None:
        ax = plt.gca()
    ax.bar(bin_centers, n, width=(bins[1] - bins[0]), color=colors)

    # Set axis limits and appearance
    ax.set_xlim(clim)
    ax.set_yticks([])

def add_colorbar_to_figure(f, cmap, clim, xxyy):
    """
    Add a colorbar to a Figure object at a specified position.

    Parameters:
    f (plt.figure): Target Figure object.
    cmap (str or Colormap): Colormap to apply.
    clim (iterable): [min, max] values for the colorbar.
    xxyy (iterable): Position in Figure coordinates (0~1) [left, bottom, width, height].

    Returns:
    matplotlib.colorbar.Colorbar: The created colorbar object.
    """
    # Add new axes for the colorbar
    cax = f.add_axes(xxyy)
    
    # Create normalization object
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    # Create ScalarMappable for color mapping
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create colorbar in the specified axes
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = f.colorbar(mappable, cax=cax, orientation=orientation)
    
    return cb

def add_colorbar_for_image(img, xxyy):
    """
    Add a colorbar to a Figure using properties (cmap, clim) from an image object.

    Parameters:
    img (matplotlib.image.AxesImage): Image object (e.g., from plt.imshow).
    xxyy (iterable): Position in Figure coordinates [left, bottom, width, height].

    Returns:
    matplotlib.colorbar.Colorbar: The created colorbar object.
    """
    fig = img.axes.figure
    cax = fig.add_axes(xxyy)
    
    # Create colorbar directly using the image object
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = fig.colorbar(img, cax=cax, orientation=orientation)
    
    return cb

def add_colorbar(artist, *, xxyy=None, cmap=None, clim=None):
    """
    Add a colorbar to a Figure, Axes, or Image object.
    Automatically infers cmap and clim from the artist if not provided.

    Parameters:
    artist (Figure, Axes, or AxesImage): Target object to add colorbar to.
    xxyy (iterable, optional): [l, b, w, h] in Figure coordinates. 
                               Default: [0.8, 0.95, 0.1, 0.02].
    cmap (str or Colormap, optional): Colormap. Inferred from artist if None.
    clim (iterable, optional): [min, max] range. Inferred from artist if None.

    Returns:
    matplotlib.colorbar.Colorbar: The created colorbar object.
    """
    # 1. Identify figure and image object based on artist type
    if isinstance(artist, matplotlib.image.AxesImage):
        target_image = artist
        f = target_image.get_figure()
    elif isinstance(artist, matplotlib.axes.Axes):
        if not artist.images:
            raise ValueError("The provided Axes object contains no images.")
        target_image = artist.images[-1]
        f = artist.get_figure()
    elif isinstance(artist, matplotlib.figure.Figure):
        if not artist.axes:
            raise ValueError("The provided Figure object has no axes.")
        ax = artist.axes[-1]
        if not ax.images:
            raise ValueError("The last Axes in the Figure contains no images.")
        target_image = ax.images[-1]
        f = artist
    else:
        raise TypeError("artist must be of type Figure, Axes, or AxesImage.")

    # 2. Infer cmap and clim if not provided
    if cmap is None:
        cmap = target_image.get_cmap()
    if clim is None:
        clim = target_image.get_clim()

    # 3. Set default position if not provided
    if xxyy is None:
        xxyy = [0.8, 0.95, 0.1, 0.02]

    # 4. Generate colorbar
    cax = f.add_axes(xxyy)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = f.colorbar(mappable, cax=cax, orientation=orientation)
    
    return cb

def subplotss(nm_rows, nm_cols, nm_sup='sup', figsize=(10, 6), **kwargs):
    """
    Wrapper function for plt.subplots with automated labeling.
    """
    nrows, ncols = len(nm_rows), len(nm_cols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    plt.suptitle(nm_sup)
    
    if nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    elif nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    
    for i, row_label in enumerate(nm_rows):
        axs[i, 0].set_ylabel(row_label)
    
    for j, col_label in enumerate(nm_cols):
        axs[0, j].set_title(col_label)
    
    return fig, axs

def _prepare_broadcastable_arg(arg, target_shape):
    """Helper function to broadcast arguments to match target subplot shape."""
    m, n = target_shape
    np_arg = np.array(arg, dtype=object)

    if np_arg.ndim == 1:
        if np_arg.shape[0] == n:
            np_arg = np_arg.reshape(1, n)
        elif np_arg.shape[0] == m:
            np_arg = np_arg.reshape(m, 1)

    try:
        return np.broadcast_to(np_arg, target_shape)
    except ValueError:
        print(f"Warning: Argument shape {np.shape(arg)} cannot broadcast to {target_shape}. Using as scalar.")
        arg_as_scalar = np.empty((1, 1), dtype=object)
        arg_as_scalar[0, 0] = np_arg
        return np.broadcast_to(arg_as_scalar, target_shape)

def broad_plot(axs, dts, draws):
    """
    Broadcast data and drawing functions across multiple axes.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    target_shape = axs.shape
    broadcasted_dts = _prepare_broadcastable_arg(dts, target_shape)
    broadcasted_draws = _prepare_broadcastable_arg(draws, target_shape)

    for i, j in np.ndindex(target_shape):
        ax = axs[i, j]
        dt = broadcasted_dts[i, j]
        draw_func = broadcasted_draws[i, j]
        # Pass axes directly to the drawing function
        draw_func(ax, dt)

def fullax(ax=None):
    """
    Expand the axes to fill the entire figure area and remove ticks.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])

def savepng(fig, fn):
    """
    Save the figure as a PNG file in the assets folder.
    """
    fig.savefig("assets/" + fn + ".png", dpi=100, bbox_inches='tight', pad_inches=0)

class CropWindow:
    """
    A class to define and manage a specific crop window (region of interest) for images or plots.
    """
    anchor_map = {
        'center': (0.5, 0.5), 'top': (0.5, 1.0), 'bottom': (0.5, 0.0),
        'left': (0.0, 0.5), 'right': (1.0, 0.5), 'top_left': (0.0, 1.0),
        'top_right': (1.0, 1.0), 'bottom_left': (0.0, 0.0), 'bottom_right': (1.0, 0.0),
    }
    
    def __init__(self):
        self._xy_01 = np.zeros((2, 2))        

    def set_xlimylim(self, xlim, ylim):
        """Sets the region by directly specifying (min, max) for x and y axes."""
        self._xy_01[0] = np.array(xlim)
        self._xy_01[1] = np.array(ylim)

    def set_yflip(self):
        """Flips the y-axis range."""
        self._xy_01[1] = np.array(self._xy_01[1])[::-1]

    def set_by_anchor(self, xy, wh, anchor='center'):
        """Sets the region centered around a specific anchor point."""
        xy = np.array(xy)
        wh = np.array(wh)
        anchor_offset = np.array(CropWindow.anchor_map[anchor])
        bl = xy - anchor_offset * wh
        self._xy_01[:, 0] = bl
        self._xy_01[:, 1] = bl + wh

    @property
    def xlim(self):
        """Returns the x-axis range (min, max)."""
        return self._xy_01[0]

    @property
    def ylim(self):
        """Returns the y-axis range (min, max)."""
        return self._xy_01[1]

    @property
    def ylim_flip(self):
        """Returns the flipped y-axis range."""
        return self._xy_01[1][::-1]

    @property
    def wh(self):
        """Returns the width and height of the region."""
        return self._xy_01[:, 1] - self._xy_01[:, 0]

    def xy(self, anchor='bottom_left'):
        """Returns coordinates of specified anchor position."""
        return self._xy_01[:, 0] + self.anchor_map[anchor] * self.wh

    def rect(self, **kwargs):
        """Returns a matplotlib.patches.Rectangle object of the current region."""
        return patches.Rectangle(self.xy(), self.wh[0], self.wh[1], fill=False, **kwargs)

    def ax_xylims(self, ax=None):
        """Applies the current region as the display range to a Matplotlib Axes."""
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim_flip)

    def ax_cropbox(self, ax=None, **kwargs):
        """Draws a rectangle box representing the region on a Matplotlib Axes."""
        if ax is None:
            ax = plt.gca()
        ax.add_patch(self.rect(**kwargs))