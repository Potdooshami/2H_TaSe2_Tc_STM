import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase


def generate_sine_wave_from_k(k_value,  arr_shape,phi=0):
  """
  Generates a 2D sine wave based on a k-value (spatial frequency vector).

  Args:
    k_value: A 2-element numpy array representing the centered k-vector
             (ky_c, kx_c) in pixel units from the FFT.
    arr_shape: A tuple (image_height, image_width) representing the shape
               of the desired output image in pixels.
    phi: Optional phase offset in radians. Defaults to 0.

  Returns:
    A 2D numpy array representing the sine wave.
  """
  # Get the centered k-vector components
  ky_c = k_value[0]
  kx_c = k_value[1]
  image_height, image_width = arr_shape
  # Create a grid of spatial coordinates (pixel indices)
  y_indices, x_indices = np.indices((image_height, image_width))

  # Calculate spatial frequencies in cycles per pixel
  spatial_freq_x_cpp = kx_c / image_width
  spatial_freq_y_cpp = ky_c / image_height

  # Construct the full argument for the sine wave, including the 2*pi factor and phase
  argument = 2 * np.pi * (spatial_freq_x_cpp * x_indices + spatial_freq_y_cpp * y_indices) + phi

  # Generate the sine wave
  sine_wave = np.sin(argument)

  return sine_wave

def kdisplacementmap(arr,k,sig):
  sw = generate_sine_wave_from_k(k,  arr.shape,0)
  cw = generate_sine_wave_from_k(k,  arr.shape,np.pi/2)
  sprod = gaussian_filter(sw*arr,sig)
  cprod = gaussian_filter(cw*arr,sig)
  xprod = cprod +sprod*1j
  return xprod




def visxprod(xprod):
  from . import auto_clim
  fig,axs = plt.subplots(2,2,figsize=(10, 10))

  #sw = generate_sine_wave_from_k(k,  aff_br.shape,0)
  tns = ['Re','Img','theta','radial']
  isbs = [0,0,1,1]
  jsbs = [0,1,0,1]
  funcs = [np.real, np.imag, np.angle, np.abs]
  cmaps = ['PuOr','vanimo','twilight','plasma']
  for isb,jsb,func,tn,cmap in zip(isbs,jsbs,funcs,tns,cmaps):
    reimg = func(xprod)
    #if tn == 'theta':
     # reimg = unwrap_phase(reimg)
    #else:
    vmin,vmax = auto_clim(reimg, method='percentile')
    im = axs[isb,jsb].imshow(reimg, cmap=cmap,vmin=vmin,vmax=vmax)
    plt.colorbar(im, ax=axs[isb,jsb])
    axs[isb,jsb].set_title(tn)
    axs[isb,jsb].set_xticks([])
    axs[isb,jsb].set_yticks([])
  return fig, axs


def wrap_phase(unwrapped_phase):
  """
  Wraps unwrapped phase data into the range of (-pi, pi].

  Args:
    unwrapped_phase: A numpy array containing the unwrapped phase data.

  Returns:
    A numpy array containing the wrapped phase data.
  """
  wrapped_phase = np.mod(unwrapped_phase + np.pi, 2 * np.pi) - np.pi
  return wrapped_phase

def calculate_unwrapped_phase(arr, k, sigma):
    """Calculates the unwrapped phase from a k-displacement map."""
    displacement_map = kdisplacementmap(arr, k, sigma)
    return unwrap_phase(np.angle(displacement_map))

def phi2Hrecover(arr_cln, ks_Latt,sig):

    angle_restores = [
    calculate_unwrapped_phase(arr_cln, k_Latt/3, sig) - 
    calculate_unwrapped_phase(arr_cln, k_Latt, sig) / 3
    for k_Latt in zip(ks_Latt)    
    ]
    return angle_restores

