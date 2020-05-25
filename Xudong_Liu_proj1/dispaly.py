import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.filters import roberts
import os

im = skio.imread(os.path.join('source_data', 'emir.tif'))

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int)

# separate color channels
b = roberts(im[:height])
g = roberts(im[height: 2*height])
r = roberts(im[2*height: 3*height])

threshold = 0.1
b[b > threshold] = 1
g[g > threshold] = 1
r[r > threshold] = 1

fname = os.path.join(os.getcwd(), 'display', 'emir_b' + '.jpg')
skio.imsave(fname, b)

fname = os.path.join(os.getcwd(), 'display', 'emir_g' + '.jpg')
skio.imsave(fname, g)

fname = os.path.join(os.getcwd(), 'display', 'emir_r' + '.jpg')
skio.imsave(fname, r)