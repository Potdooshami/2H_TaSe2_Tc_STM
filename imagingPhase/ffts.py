from scipy.fft import fft2, fftshift
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def get_magnitude_spectrum(arr_cln):
  fft_result = fft2(arr_cln)
  fft_result_shifted = fftshift(fft_result)
  magnitude_spectrum = np.abs(fft_result_shifted)
  return magnitude_spectrum

def fft2show(arr_cln,vmin,vmax):
  fft_result = fft2(arr_cln)
  fft_result_shifted = fftshift(fft_result)
  magnitude_spectrum = np.abs(fft_result_shifted)
  plt.figure(figsize=(10, 10))
  plt.imshow(magnitude_spectrum, cmap='gray', vmin=vmin, vmax=vmax)
  plt.title('Magnitude Spectrum of 2D FFT')
  return  magnitude_spectrum

def fft2pkfnd(fft2abs,threshold,choose):
  coordinates = peak_local_max(fft2abs, min_distance=100, threshold_abs = threshold)
  plt.scatter(coordinates[:, 1], coordinates[:, 0], s=50, facecolors='none', edgecolors='r')
  for ipeak in range(coordinates.shape[0]):
    plt.text(coordinates[ipeak,1],coordinates[ipeak,0],str(ipeak), color='g')
  coordinates_choose = coordinates[choose,:]
  print(coordinates_choose)
  print(coordinates_choose.shape[0])
  for ipeak in range(coordinates_choose.shape[0]):
    plt.text(coordinates_choose[ipeak,1],coordinates_choose[ipeak,0],str(ipeak), color='b',size= 20)
    pk_all = coordinates - (np.array(fft2abs.shape)/2)
    pk_choose = pk_all[choose,:]
  return pk_choose,pk_all

def get_line_profile(img, p1, p2, num_points):
    """
    Get the line profile of a 2D image between two points p1 and p2.
    
    Args:
        img (np.ndarray): The 2D image.
        p1 (tuple): The (y, x) coordinates of the start point.
        p2 (tuple): The (y, x) coordinates of the end point.
        num_points (int): The number of points for interpolation.
        
    Returns:
        np.ndarray: The 1D array of intensity values along the line.
    """
    # Generate coordinates along the line
    y_coords = np.linspace(p1[0], p2[0], num_points)
    x_coords = np.linspace(p1[1], p2[1], num_points)
    
    # `map_coordinates` expects coordinates in (ndim, n_points) format
    coords = np.vstack((y_coords, x_coords))
    
    # Interpolate the values from the image
    profile = map_coordinates(img, coords, order=1)
    
    return profile

def autocorr2d_fft(arr):
    f = np.fft.fft2(arr)
    psd = f * np.conj(f)   # power spectral density
    result = np.fft.ifft2(psd).real
    return np.fft.fftshift(result)  # 중앙 정렬