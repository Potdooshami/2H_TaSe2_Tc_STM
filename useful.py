def tickoff():
  """
  Remove ticks from the current axes.
  """
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
        The new minimum value for the color mapping. If None, the current vmin is kept.
    vmax : float, optional
        The new maximum value for the color mapping. If None, the current vmax is kept.
    """
    if vmin is not None:
        image_object.set_clim(vmin=vmin)
    if vmax is not None:
        image_object.set_clim(vmax=vmax)

def auto_clim(data, method='percentile', lower=1, upper=99):
    """
    data: 2D array
    method: 'percentile' or 'std'
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
  Automatically adjust the color limits of the current image to the 1% and 99%
  """
  ic = gci()
  foo = ic.get_array()
  aclim =  auto_clim(foo, method='percentile', lower=1, upper=99)
  change_image_range(ic, aclim[0], aclim[1])


def plot_result(image, background):
    """
    Plot the original image, background, and the result of image - background side by side.
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
    return fig,ax