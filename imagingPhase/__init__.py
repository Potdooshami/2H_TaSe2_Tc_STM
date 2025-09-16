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

  """
  Automatically adjust the color limits of the current image to the 1% and 99%
  """
  ic = gci()
  foo = ic.get_array()
  aclim =  auto_clim(foo, method='percentile', lower=1, upper=99)
  change_image_range(ic, aclim[0], aclim[1])