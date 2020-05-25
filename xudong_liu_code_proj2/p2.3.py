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
		        
def gussian_2d(size, sigma):
    gussian = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    return np.outer(gussian, gussian)

def laplacian_filter(size, sigma):
    impulse = np.zeros((size,size))
    impulse[int((size - 1)/2) ,int((size - 1)/2)] = 1
    return impulse - gussian_2d(size, sigma)

def conv_2d(img, filter):
    if img.shape[2]==3:
        img1 = scipy.signal.convolve2d(img[:,:,0], filter, boundary='symm', mode='same')
        img2 = scipy.signal.convolve2d(img[:,:,1], filter, boundary='symm', mode='same')
        img3 = scipy.signal.convolve2d(img[:,:,2], filter, boundary='symm', mode='same')
        out = np.dstack((img1, img2, img3))
    else:
        out = scipy.signal.convolve2d(img, filter, boundary='symm', mode='same')
    return out

path = 'p2.3'
mkdir(path)
img_name = 'monalisa.jpg'
out_name = img_name.split('.')[0] + '_stack.jpg'

img = io.imread(img_name)
num_level = 8
gussian = img
# laplacian = img
for level in range(num_level):
    laplacian = conv_2d(gussian, laplacian_filter(19, 6))
    gussian = conv_2d(gussian, gussian_2d(19, 6))
    io.imsave(os.path.join(path, 'gussian' + str(level) + out_name), gussian)
    if level < num_level - 1:
        io.imsave(os.path.join(path, 'laplacian' + str(level) + out_name), laplacian)
    else:
        io.imsave(os.path.join(path, 'laplacian' + str(level) + out_name), gussian)

