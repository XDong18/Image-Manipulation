import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.ndimage.interpolation
import cv2
import scipy.signal
import math
import os


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

im_name = 'rail.jpg'
img = io.imread(im_name, as_gray=True)
angles_list = np.arange(-4, 4, 0.25)

dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])
gussian = cv2.getGaussianKernel(ksize=5, sigma=1)
gussian2d = np.outer(gussian, gussian)
dx_gu = scipy.signal.convolve2d(gussian2d, dx, boundary='fill', mode='full')
dy_gu = scipy.signal.convolve2d(dy, gussian2d, boundary='fill', mode='full')
h, w = img.shape

path = 'p1.3'
img_path = im_name.split('.')[0]
mkdir(path)
mkdir(os.path.join(path, img_path))
img = io.imread(im_name)
img_rotated = scipy.ndimage.interpolation.rotate(img, 2.5, reshape=False)
io.imsave(os.path.join(path, img_path, 'result' + 'rotated.jpg'), img_rotated)

# for angle in angles_list:
#     img_rotated = scipy.ndimage.interpolation.rotate(img, angle, reshape=False)
#     window = img_rotated[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
#     bl_dx_img = scipy.signal.convolve2d(window, dx_gu, boundary='fill', mode='same')
#     bl_dy_img = scipy.signal.convolve2d(window, dy_gu, boundary='fill', mode='same')
#     angle_map = np.arctan(bl_dy_img / bl_dx_img) / math.pi * 180
#     angle_mapf = np.reshape(angle_map,(-1))
#     plt.hist(angle_mapf, 1000)
#     plt.axis([-90, 90, 0, 100])
#     plt.plot([0, 0], [0, 100], 'k-', lw=1,dashes=[2, 2])
#     plt.savefig(os.path.join(path, img_path, str(round(angle, 1)) + 'rotated.jpg'))
#     plt.close('all')