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
    import matplotlib.pyplot as plt
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
    import numpy as np
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
  import matplotlib.pyplot as plt

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

def subtabs(figs, titles=None):
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

def subtabSr(arrSr):
    '''
    arrSr: pd.Series of matplotlib.figure.Figure
    '''
    import matplotlib.pyplot as plt
    def figimshow(arr):        
        fig = plt.figure(figsize=(20,20))
        plt.imshow(arr,cmap='afmhot')
        try:
            auto199()
        except:
            pass
        # tickoff()
        plt.close(fig)        
        return fig
    figs = list(map(figimshow,list(arrSr.values)))
    titles = arrSr.index.to_list()
    tabs = subtabs(figs,titles)
    return tabs
    