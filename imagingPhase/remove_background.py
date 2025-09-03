import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_opening, disk, binary_dilation
from skimage.restoration import inpaint

def ceilcut(image,threshold):
  """
    Clip values in the image above the specified threshold.
  """
  imagec = image.copy()
  imagec[image>threshold] = threshold
  return imagec

def replace_with_average(img_original, bw):
  """
  Replaces the value of specific part of image as average.

  Args:
    img_original: (n,m) float array, the original image.
    bw: (n,m) boolean array, the mask where True indicates the part to be replaced.

  Returns:
    img_cleaned: (n,m) float array, the image with the specified part replaced by the average.
  """
  # Calculate the average of the values in img_original where bw is False
  average_false_part = np.mean(img_original[~bw])

  # Create a copy of the original image
  img_cleaned = img_original.copy()

  # Replace the values in img_cleaned where bw is True with the calculated average
  img_cleaned[bw] = average_false_part

  return img_cleaned