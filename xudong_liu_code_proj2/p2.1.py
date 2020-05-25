import cv2
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

def gaussian_2d(size, sigma):
    gaussian = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    return np.outer(gaussian, gaussian)

def laplacian_filter(size, sigma):
    impulse = np.zeros((size,size))
    impulse[int((size - 1)/2) ,int((size - 1)/2)] = 1
    return impulse - gaussian_2d(size, sigma)

def conv_2d(img, filter):
    if img.shape[2]==3:
        img1 = scipy.signal.convolve2d(img[:,:,0], filter, boundary='symm', mode='same')
        img2 = scipy.signal.convolve2d(img[:,:,1], filter, boundary='symm', mode='same')
        img3 = scipy.signal.convolve2d(img[:,:,2], filter, boundary='symm', mode='same')
        out = np.dstack((img1, img2, img3))
    else:
        out = scipy.signal.convolve2d(img, filter, boundary='symm', mode='same')
    return out

im_name = 'flower.jpg'
img = io.imread(im_name)
blur_img = conv_2d(img, gaussian_2d(7, 2))
# sharp_img = conv_2d(img, laplacian_filter(3, 1))
sharp_img = conv_2d(blur_img, laplacian_filter(3, 1))

# final = (img * 0.5 + sharp_img * 0.5)
final = (blur_img * 0.5 + sharp_img * 0.5)

path ='p2.1'
mkdir(path)
io.imsave(os.path.join(path, 'blured_' + im_name), blur_img)
io.imsave(os.path.join(path, 'sharped_' + im_name), final)