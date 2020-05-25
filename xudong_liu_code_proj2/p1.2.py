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

dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])

img = io.imread('cameraman.png', as_gray=True)

gussian2d = gussian_2d(5, 1)
bler_img = scipy.signal.convolve2d(img, gussian2d, boundary='symm', mode='same')

dx_gu = scipy.signal.convolve2d(gussian2d, dx, boundary='fill', mode='full')
dy_gu = scipy.signal.convolve2d(dy, gussian2d, boundary='fill', mode='full')

bl_dx_img = scipy.signal.convolve2d(img, dx_gu, boundary='symm', mode='same')
bl_dy_img = scipy.signal.convolve2d(img, dy_gu, boundary='symm', mode='same')
bl_grad_img = np.power(np.power(bl_dx_img, 2) + np.power(bl_dy_img, 2), 0.5)

edge_img = np.zeros(bl_grad_img.shape)
th = 25/255
edge_img[np.where(bl_grad_img>=th)] = 255
edge_img[np.where(bl_grad_img<th)] = 0

path = 'p1.2'
mkdir(path)

io.imsave(os.path.join(path, 'bler.jpg'), bler_img)
io.imsave(os.path.join(path, 'bl_dx.jpg'), bl_dx_img)
io.imsave(os.path.join(path, 'bl_dy.jpg'), bl_dy_img)
io.imsave(os.path.join(path, 'grad.jpg'), bl_grad_img)
io.imsave(os.path.join(path, 'bl_edge.jpg'), edge_img)