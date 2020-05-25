import matplotlib.pyplot as plt
from align_image_code import align_images
import skimage
from skimage import io 
import numpy as np 
import os
import cv2
import scipy.signal


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

def fourier(input):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(input))))

    average = 0.5 * low_img1 + 0.5 * high_img2