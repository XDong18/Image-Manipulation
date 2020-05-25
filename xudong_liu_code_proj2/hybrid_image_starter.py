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
    if input.shape[2] == 3:
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(input[:,:,1]))))
    else:
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(input))))

def gaussian_2d(size, sigma):
    gaussian = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    return np.outer(gaussian, gaussian)

def laplacian_filter(size, sigma):
    impulse = np.zeros((size,size))
    impulse[int((size - 1)/2) ,int((size - 1)/2)] = 1
    return impulse - gaussian_2d(size, sigma)

def conv_2d(img, filter):
    # print(img.shape)
    if img.shape[2]==3:
        img1 = scipy.signal.convolve2d(img[:,:,0], filter, boundary='symm', mode='same')
        img2 = scipy.signal.convolve2d(img[:,:,1], filter, boundary='symm', mode='same')
        img3 = scipy.signal.convolve2d(img[:,:,2], filter, boundary='symm', mode='same')
        out = np.dstack((img1, img2, img3))
    else:
        out = scipy.signal.convolve2d(img, filter, boundary='symm', mode='same')
    return out

def hybrid_image(im1, im2, sigma1, sigma2):
    low_img = conv_2d(im1, gaussian_2d(29, sigma1))
    io.imsave(os.path.join(path, name1.split('.')[0] + '_low_fft.jpg'), fourier(low_img))
    high_img = conv_2d(im2, laplacian_filter(29, sigma2))
    io.imsave(os.path.join(path, name2.split('.')[0] + '_high_fft.jpg'), fourier(high_img))
    io.imsave(os.path.join(path, name1.split('.')[0] + name2.split('.')[0] + '_hybrid.jpg'), fourier(0.5 * low_img + 0.5 * high_img))

    return 0.5 * low_img + 0.5 * high_img

def pyramids(img, N):
    gaussian = img
    for level in range(N):
        laplacian = conv_2d(gaussian, laplacian_filter(19, 6))
        gaussian = conv_2d(gaussian, gaussian_2d(19, 6))
        io.imsave(os.path.join(path, 'gaussian' + str(level) + out_name), gaussian)
        if level < N - 1:
            io.imsave(os.path.join(path, 'laplacian' + str(level) + out_name), laplacian)
        else:
            io.imsave(os.path.join(path, 'laplacian' + str(level) + out_name), gaussian)

# First load images
path = 'p2.2'
mkdir(path)
out_name = 'man_puppy.jpg'
name1 = 'man.jpg'
name2 = 'puppy.jpg'
# high sf
im1 = plt.imread(name1)/255.
io.imsave(os.path.join(path, name1.split('.')[0] + '_fft.jpg'), fourier(im1))
# print(im1.shape)
# im1 = np.expand_dims(im1, axis=2) 
# low sf
im2 = plt.imread(name2)/255.
io.imsave(os.path.join(path, name2.split('.')[0] + '_fft.jpg'), fourier(im2))

# im2 = np.expand_dims(im2, axis=2) 

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)
# plt.imsave('im1.jpg', im1_aligned)
# plt.imsave('im2.jpg', im2_aligned)
# You will provide the code below. Sigma1 and sigma2 are arbitrary 
# cutoff values for the high and low frequencies

sigma1 = 9
sigma2 = 7
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
io.imsave(os.path.join(path, out_name), hybrid)

plt.imshow(hybrid)
plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)