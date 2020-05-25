import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.signal
import os


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

img = io.imread('cameraman.png', as_gray=True)

dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])

dx_img = scipy.signal.convolve2d(img, dx, boundary='symm', mode='same')
dy_img = scipy.signal.convolve2d(img, dy, boundary='symm', mode='same')
grad_img = np.power(np.power(dx_img, 2) + np.power(dy_img, 2), 0.5)

edge_img = np.zeros(grad_img.shape)
 
th = 75/255
edge_img[np.where(grad_img>=th)] = 255
edge_img[np.where(grad_img<th)] = 0

path = 'p1.1'
mkdir(path)
io.imsave(os.path.join(path, 'dx.jpg'), dx_img)
io.imsave(os.path.join(path, 'dy.jpg'), dy_img)
io.imsave(os.path.join(path, 'grad.jpg'), grad_img)
io.imsave(os.path.join(path, 'edge.jpg'), edge_img)



