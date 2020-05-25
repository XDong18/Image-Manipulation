import cv2
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os
from skimage.transform import resize


def mkdir(path):
	folder = os.path.exists(path)
 
	if not folder:                  
		os.makedirs(path)          
		print ("---  new folder...")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")        

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

path = 'p2.4'
mkdir(path)

name1 = 'apple.jpeg'
name2 = 'orange.jpeg'
# high sf
im1 = io.imread(name1)
im2 = io.imread(name2)
im2 = resize(im2, (im1.shape[0], im1.shape[1]), anti_aliasing=True)
h, w, _ = im1.shape

result = np.zeros(im1.shape)
result2 = np.zeros(im1.shape)
def blend(img, width, l_r):
    h, w, c = img.shape
    
    if l_r==True:
        mid = np.arange(1, 0, -1/width)
        mid = mid.reshape((1, mid.size))
        left_num = int((h-width)/2)
        left = np.ones((1, left_num))
        right =  np.zeros((1, w - width - left_num))
        mask = np.concatenate((left, mid, right), axis=1)
        mask = np.repeat(mask, repeats=h, axis=0)
        mask = np.expand_dims(mask, axis=2) 
        mask = np.repeat(mask, repeats=c, axis=2)
        # mask = np.zeros(img.shape)
        # mask[:, 0:int((h-w)/2), :] = 1
        # mask[:, int((h-w)/2):int((h-w)/2)+w, :] = np.arange(1, 0, -1/w)
    else:
        mid = np.arange(0, 1, 1/width)
        mid = mid.reshape((1, mid.size))
        left_num = int((h-width)/2)
        left = np.zeros((1, left_num))
        right =  np.ones((1, w - width - left_num))
        mask = np.concatenate((left, mid, right), axis=1)
        mask = np.repeat(mask, repeats=h, axis=0)
        mask = np.expand_dims(mask, axis=2) 
        mask = np.repeat(mask, repeats=c, axis=2)
    return img * mask

# img_name = 'monalisa.jpg'
# out_name = img_name.split('.')[0] + '_stack.jpg'

# img = io.imread(img_name)
num_level = 8
begin = 1
# laplacian = img
gussian = im1

for level in range(num_level):
    laplacian = conv_2d(gussian, laplacian_filter(19, 6))
    gussian = conv_2d(gussian, gussian_2d(19, 6))
    #result = result +  blend(laplacian, begin * pow(2,(level + 1)), True)
    # io.imsave(os.path.join(path, 'gussian' + str(level) + out_name), gussian)
    if level < num_level - 1:
        result = result +  blend(laplacian, begin * pow(2,(level + 1)), True)
        io.imsave(os.path.join(path, 'laplacian' + str(level) + name1), blend(laplacian, begin * pow(2,(level + 1)), True))
    else:
        result = result +  blend(gussian, begin * pow(2,(level + 1)), True)
        io.imsave(os.path.join(path, 'laplacian' + str(level) + name1), blend(gussian, begin * pow(2,(level + 1)), True))
io.imsave(os.path.join(path,'temp_ao.jpg'), result)

gussian = im2
for level in range(num_level):
    laplacian = conv_2d(gussian, laplacian_filter(19, 6))
    gussian = conv_2d(gussian, gussian_2d(19, 6))
    #result = result +  blend(laplacian, begin * pow(2,(level + 1)), False)
    # io.imsave(os.path.join(path, 'gussian' + str(level) + out_name), gussian)
    if level < num_level - 1:
        result2 = result2 +  blend(laplacian, begin * pow(2,(level + 1)), False)
        io.imsave(os.path.join(path, 'laplacian' + str(level) + name2), blend(laplacian, begin * pow(2,(level + 1)), False))
    else:
        result2 = result2 +  blend(gussian, begin * pow(2,(level + 1)), False)
        io.imsave(os.path.join(path, 'laplacian' + str(level) + name2), blend(gussian, begin * pow(2,(level + 1)), False))

io.imsave(os.path.join(path, 'apple_orange.jpg'), result + result2)
io.imsave(os.path.join(path, '_orange2.jpg'), result2)
