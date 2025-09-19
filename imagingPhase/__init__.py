Sucessfully__init__ = 10
print("imagingPhase package loaded")
def init_function():
    return "This is the init function in imagingPhase package"


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
    ic = gci()
    foo = ic.get_array()
    aclim =  auto_clim(foo, method='percentile', lower=1, upper=99)
    change_image_range(ic, aclim[0], aclim[1])

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