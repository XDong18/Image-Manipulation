# CS194-26 (CS294-26): Project 1 starter Python code

'''
align images with edge information
(for .tif images, use multiscale pyramid version; for .jpg images, use multiscale pyramid version)
@Author: Xudong Liu
@Email: liuxd@berkeley.edu
'''

import time
import os
from utils import *


image_list = ['emir.tif', 'self_portrait.tif']

for imname in image_list:
    print(imname)
    start = time.time()

    # name of the input file
    # imname = 'three_generations.tif'

    # read in the image
    im = skio.imread(os.path.join('source_data', imname))

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # multiscale pyramid version that works on the large images
    if(imname.split('.')[1]=='tif'):
        # initialize pyramid and target classes
        g_pyramid = pyramid_edge(g)
        r_pyramid = pyramid_edge(r)
        b_target = target(b)

        # align 1/8 image
        g_pyramid.py_align_8(b)
        r_pyramid.py_align_8(b)

        # align 1/4 image
        g_pyramid.py_align_4(b)
        r_pyramid.py_align_4(b)

        # align 1/2 image
        g_pyramid.py_align_2(b)
        r_pyramid.py_align_2(b)

        # align whole image
        g_pyramid.py_align(b)
        r_pyramid.py_align(b)

        # result
        im_out = np.dstack([r_pyramid.im, g_pyramid.im, b_target.im])

        end = time.time()

        g_pyramid.info['time'] = end - start
        r_pyramid.info['time'] = end - start
        g_pyramid.save_info(os.path.join('info', imname.split('.')[0] + '_g.json'))
        r_pyramid.save_info(os.path.join('info', imname.split('.')[0] + '_r.json'))

        
    # c that works on the small images
    else:
        ag, info_g = align_single(g, b)
        ar, info_r = align_single(r, b)

        # create a color image
        im_out = np.dstack([ar, ag, b])

        end = time.time()

        info_g['time'] = end - start
        info_r['time'] = end - start

        with(open(os.path.join('info', imname.split('.')[0] + '_g.json'), 'w')) as f:
            json.dump(info_g, f)

        with(open(os.path.join('info', imname.split('.')[0] + '_r.json'), 'w')) as f:
            json.dump(info_r, f)

    print("time cost is %ss." % (end - start))

    # display the image
    # skio.imshow(im_out)
    # skio.show()

    # save the image
    fname = os.path.join(os.getcwd(), 'out_path', 'bells_whistles', imname.split('.')[0] + '.jpg')
    skio.imsave(fname, im_out)

