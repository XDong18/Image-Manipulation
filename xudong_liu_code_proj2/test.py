import cv2
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os
from skimage.transform import resize

im1 = io.imread('C://Users//Administrator//cs194//proj2//p2.4//_orange2.jpg')
im2 = io.imread("C://Users//Administrator//cs194//proj2//p2.4//temp_ao.jpg")
print(im1)
print(im2)
img = 0.5 * im1 + 0.5 * im2
io.imsave('test2.jpg',img)
# io.show()