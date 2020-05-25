
'''
utils
@Author: Xudong Liu
@Email: liuxd@berkeley.edu
'''

import numpy as np
import skimage as sk
import skimage.io as skio
import json
from skimage.filters import roberts


def align_single(im_a, im_b, shift_rate=0.05, window_rate=0.45):
    '''
    Function: Align im_a with respect to im_b(single scale image version)
    Return: an aligned image array

    im_a: image to be aligned
    im_b: the target image
    shift_rate: shift range = (h, w) * shift_rate
    window_rate: window range = (h, w) * window_rate;
                 windows are the areas of given images used to align im_a and inm_b
    '''
    # shapes of im_a and im_b are the same
    h, w = im_a.shape

    info = {}
    
    # shift h,w
    shift_h = np.floor(shift_rate * h).astype(np.int)
    shift_w = np.floor(shift_rate * h).astype(np.int)

    # window boundarys used to align the images
    window_h_up = np.floor((0.5 - window_rate) * h).astype(np.int)
    window_h_down = np.floor((0.5 + window_rate) * h).astype(np.int)
    window_w_left = np.floor((0.5 - window_rate) * w).astype(np.int)
    window_w_right = np.floor((0.5 + window_rate) * w).astype(np.int)

    # ncc matrix stores the ncc number of each offset
    ncc_matrix = np.ones((2 * shift_h, 2 * shift_w)) * -1
    window_b = im_b[window_h_up:window_h_down, window_w_left:window_w_right]
    window_b = window_b - np.mean(window_b)
    # normalization
    norm_b = window_b / np.linalg.norm(window_b)

    # h, w offsets
    for index_h in list(range(2 * shift_h)):
        for index_w in list(range(2 * shift_w)):
            # roll a
            roll_a = np.roll(im_a, (index_h - shift_h, index_w - shift_w), (0, 1))
            window_a = roll_a[window_h_up:window_h_down, window_w_left:window_w_right]
            window_a = window_a - np.mean(window_a)
            # normalization
            norm_a = window_a / np.linalg.norm(window_a)
            # calculate ncc
            ncc = sum(sum(norm_a * norm_b))
            ncc_matrix[index_h, index_w] = ncc

    # the offset with the biggest ncc
    shift_y, shift_x = np.unravel_index(np.argmax(ncc_matrix),ncc_matrix.shape)

    info['h'] = int(shift_y - shift_h)
    info['w'] = int(shift_x - shift_w)

    # return the aligned image
    return np.roll(im_a, (shift_y - shift_h, shift_x - shift_w), (0, 1)), info

def align(im_a, im_b, shift_rate=0.05, window_rate=0.25):
    '''
    Function: Align im_a with respect to im_b(multiscale pyramid version)
    Return: a tuple: (offset_h, offset_w)

    im_a: image to be aligned
    im_b: the target image
    shift_rate: shift range = (h, w) * shift_rate
    window_rate: window range = (h, w) * window_rate;
                 windows are the areas of given images used to align im_a and inm_b
    '''
    # shapes of im_a and im_b are the same
    h, w = im_a.shape

    # shift h,w
    shift_h = np.floor(shift_rate * h).astype(np.int)
    shift_w = np.floor(shift_rate * h).astype(np.int)

    # window boundarys used to align the images
    window_h_up = np.floor((0.5 - window_rate) * h).astype(np.int)
    window_h_down = np.floor((0.5 + window_rate) * h).astype(np.int)
    window_w_left = np.floor((0.5 - window_rate) * w).astype(np.int)
    window_w_right = np.floor((0.5 + window_rate) * w).astype(np.int)

    # ncc matrix stores the ncc number of each offset
    ncc_matrix = np.ones((2 * shift_h, 2 * shift_w)) * -1
    window_b = im_b[window_h_up:window_h_down, window_w_left:window_w_right]
    window_b = window_b - np.mean(window_b)
    # normalization
    norm_b = window_b / np.linalg.norm(window_b)

    # h, w offsets
    for index_h in list(range(2 * shift_h)):
        for index_w in list(range(2 * shift_w)):
            # roll a
            roll_a = np.roll(im_a, (index_h - shift_h, index_w - shift_w), (0, 1))
            window_a = roll_a[window_h_up:window_h_down, window_w_left:window_w_right]
            window_a = window_a - np.mean(window_a)
            # normalization
            norm_a = window_a / np.linalg.norm(window_a)
            # calculate ncc
            ncc = sum(sum(norm_b * norm_a))
            ncc_matrix[index_h, index_w] = ncc
            
    # return the offset with the biggest ncc
    return np.unravel_index(np.argmax(ncc_matrix),ncc_matrix.shape)


class pyramid():
    '''
    used for multi-scale aligning
    '''
    def __init__(self, im):
        '''
        im: a image array
        '''
        self.im = im

        # 1/2, 1/4, 1/8 iamges
        self.im_2 = sk.transform.rescale(self.im, 0.5, anti_aliasing=False)
        self.im_4 = sk.transform.rescale(self.im, 1/4, anti_aliasing=False)
        self.im_8 = sk.transform.rescale(self.im, 1/8, anti_aliasing=False)

        # 1/2, 1/4, 1/8 h and w
        self.h = [self.im.shape[0], self.im_2.shape[0], self.im_4.shape[0], self.im_8.shape[0]]
        self.w = [self.im.shape[1], self.im_2.shape[1], self.im_4.shape[1], self.im_8.shape[1]]
        self.shift_rate = np.array([0.05/32, 0.05/16, 0.05/8, 0.05]).astype(np.float)

        # shift rates
        self.shift_h = np.floor(self.shift_rate * self.h).astype(np.int)
        self.shift_w = np.floor(self.shift_rate * self.w).astype(np.int)

        self.target = None

        self.info = {}
        self.info['h'] = 0
        self.info['w'] = 0

    def py_align_8(self, target_im):
        '''
        Function: align 1/8 iamge and update 1/8, 1/4, 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/8 image
        shift_y, shift_x = align(self.im_8, self.target.im_8, self.shift_rate[3], 0.45)
        self.info['h'] += (shift_y - self.shift_h[3]) * 8
        self.info['w'] += (shift_x - self.shift_w[3]) * 8

        # update 1/8, 1/4, 1/2, whole images
        self.im_8 = np.roll(self.im_8, (shift_y - self.shift_h[3], shift_x - self.shift_w[3]), (0, 1))
        self.im_4 = np.roll(self.im_4, (2 * (shift_y - self.shift_h[3]), 2 * (shift_x - self.shift_w[3])), (0, 1))
        self.im_2 = np.roll(self.im_2, (4 * (shift_y - self.shift_h[3]), 4 * (shift_x - self.shift_w[3])), (0, 1))
        self.im = np.roll(self.im, (8 * (shift_y - self.shift_h[3]), 8 * (shift_x - self.shift_w[3])), (0, 1))
        print('1/8 image is aligned.')

    def py_align_4(self, target_im):
        '''
        Function: align 1/4 iamge and update 1/4, 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/4 image
        shift_y, shift_x = align(self.im_4, self.target.im_4, self.shift_rate[2], 0.35)
        self.info['h'] += (shift_y - self.shift_h[2]) * 4
        self.info['w'] += (shift_x - self.shift_w[2]) * 4

        # update 1/4, 1/2, whole images
        self.im_4 = np.roll(self.im_4, (shift_y - self.shift_h[2], shift_x - self.shift_w[2]), (0, 1))
        self.im_2 = np.roll(self.im_2, (2 * (shift_y - self.shift_h[2]), 2 * (shift_x - self.shift_w[2])), (0, 1))
        self.im = np.roll(self.im, (4 * (shift_y - self.shift_h[2]), 4 * (shift_x - self.shift_w[2])), (0, 1))
        print('1/4 image is aligned.')

    def py_align_2(self, target_im):
        '''
        Function: align 1/2 iamge and update 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/2 image
        shift_y, shift_x = align(self.im_2, self.target.im_2, self.shift_rate[1], 0.35)
        self.info['h'] += (shift_y - self.shift_h[1]) * 2
        self.info['w'] += (shift_x - self.shift_w[2]) * 2

        # update 1/2, whole images
        self.im_2 = np.roll(self.im_2, (shift_y - self.shift_h[1], shift_x - self.shift_w[2]), (0, 1))
        self.im = np.roll(self.im, (2 * (shift_y - self.shift_h[1]), 2 * (shift_x - self.shift_w[1])), (0, 1))
        print('1/2 image is aligned.')
    
    def py_align(self, target_im):
        '''
        Function: align whole iamge and whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align whole image
        shift_y, shift_x = align(self.im, self.target.im, self.shift_rate[0], 0.35)
        self.info['h'] += (shift_y - self.shift_h[0])
        self.info['w'] += (shift_x - self.shift_w[0])

        # update whole image
        self.im = np.roll(self.im, (shift_y - self.shift_h[0], shift_x - self.shift_w[0]), (0, 1))
        print('whole image is aligned.')
    
    def save_info(self, file_name):
        self.info['h'] = int(self.info['h'])
        self.info['w'] = int(self.info['w'])
        with open(file_name, 'w') as f:
            json.dump(self.info, f)


class target():
    '''
    used for multi-scale aligning
    '''
    def __init__(self, im):
        '''
        im: target image array
        '''
        self.im = im
        # 1/2, 1/4, 1/8 iamges
        self.im_2 = sk.transform.rescale(self.im, 0.5, anti_aliasing=False)
        self.im_4 = sk.transform.rescale(self.im, 1/4, anti_aliasing=False)
        self.im_8 = sk.transform.rescale(self.im, 1/8, anti_aliasing=False)


def align_single_edge(im_a, im_b, shift_rate=0.05, window_rate=0.45):
    '''
    Function: Align im_a with respect to im_b using edge information(single scale image version)
    Return: an aligned image array

    im_a: image to be aligned
    im_b: the target image
    shift_rate: shift range = (h, w) * shift_rate
    window_rate: window range = (h, w) * window_rate;
                 windows are the areas of given images used to align im_a and inm_b
    '''
    # shapes of im_a and im_b are the same
    h, w = im_a.shape

    # original image
    im_a_real = im_a

    # egde filter
    im_a = roberts(im_a)
    im_b = roberts(im_b)

    info = {}
    
    # shift h,w
    shift_h = np.floor(shift_rate * h).astype(np.int)
    shift_w = np.floor(shift_rate * h).astype(np.int)

    # window boundarys used to align the images
    window_h_up = np.floor((0.5 - window_rate) * h).astype(np.int)
    window_h_down = np.floor((0.5 + window_rate) * h).astype(np.int)
    window_w_left = np.floor((0.5 - window_rate) * w).astype(np.int)
    window_w_right = np.floor((0.5 + window_rate) * w).astype(np.int)

    # ncc matrix stores the ncc number of each offset
    ncc_matrix = np.ones((2 * shift_h, 2 * shift_w)) * -1
    window_b = im_b[window_h_up:window_h_down, window_w_left:window_w_right]
    window_b = window_b - np.mean(window_b)
    # normalization
    norm_b = window_b / np.linalg.norm(window_b)

    # h, w offsets
    for index_h in list(range(2 * shift_h)):
        for index_w in list(range(2 * shift_w)):
            # roll a
            roll_a = np.roll(im_a, (index_h - shift_h, index_w - shift_w), (0, 1))
            window_a = roll_a[window_h_up:window_h_down, window_w_left:window_w_right]
            window_a = window_a - np.mean(window_a)
            # normalization
            norm_a = window_a / np.linalg.norm(window_a)
            # calculate ncc
            ncc = sum(sum(norm_a * norm_b))
            ncc_matrix[index_h, index_w] = ncc

    # the offset with the biggest ncc
    shift_y, shift_x = np.unravel_index(np.argmax(ncc_matrix),ncc_matrix.shape)

    info['h'] = int(shift_y - shift_h)
    info['w'] = int(shift_x - shift_w)

    # return the aligned image
    return np.roll(im_a_real, (shift_y - shift_h, shift_x - shift_w), (0, 1)), info

class pyramid_edge():
    '''
    used for multi-scale aligning using edge information
    '''
    def __init__(self, im):
        '''
        im: a image array
        '''
        self.im = im

        # 1/2, 1/4, 1/8 iamges
        self.im_2 = sk.transform.rescale(self.im, 0.5, anti_aliasing=False)
        self.im_4 = sk.transform.rescale(self.im, 1/4, anti_aliasing=False)
        self.im_8 = sk.transform.rescale(self.im, 1/8, anti_aliasing=False)

        # 1/2, 1/4, 1/8 h and w
        self.h = [self.im.shape[0], self.im_2.shape[0], self.im_4.shape[0], self.im_8.shape[0]]
        self.w = [self.im.shape[1], self.im_2.shape[1], self.im_4.shape[1], self.im_8.shape[1]]
        self.shift_rate = np.array([0.05/32, 0.05/16, 0.05/8, 0.05]).astype(np.float)

        # shift rates
        self.shift_h = np.floor(self.shift_rate * self.h).astype(np.int)
        self.shift_w = np.floor(self.shift_rate * self.w).astype(np.int)

        self.target = None

        self.info = {}
        self.info['h'] = 0
        self.info['w'] = 0

    def py_align_8(self, target_im):
        '''
        Function: align 1/8 iamge and update 1/8, 1/4, 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/8 image
        # using roberts filter to get edge information
        shift_y, shift_x = align(roberts(self.im_8), roberts(self.target.im_8), self.shift_rate[3], 0.45)
        self.info['h'] += (shift_y - self.shift_h[3]) * 8
        self.info['w'] += (shift_x - self.shift_w[3]) * 8

        # update 1/8, 1/4, 1/2, whole images
        self.im_8 = np.roll(self.im_8, (shift_y - self.shift_h[3], shift_x - self.shift_w[3]), (0, 1))
        self.im_4 = np.roll(self.im_4, (2 * (shift_y - self.shift_h[3]), 2 * (shift_x - self.shift_w[3])), (0, 1))
        self.im_2 = np.roll(self.im_2, (4 * (shift_y - self.shift_h[3]), 4 * (shift_x - self.shift_w[3])), (0, 1))
        self.im = np.roll(self.im, (8 * (shift_y - self.shift_h[3]), 8 * (shift_x - self.shift_w[3])), (0, 1))
        print('1/8 image is aligned.')

    def py_align_4(self, target_im):
        '''
        Function: align 1/4 iamge and update 1/4, 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/4 image
        # using roberts filter to get edge information
        shift_y, shift_x = align(roberts(self.im_4), roberts(self.target.im_4), self.shift_rate[2], 0.35)
        self.info['h'] += (shift_y - self.shift_h[2]) * 4
        self.info['w'] += (shift_x - self.shift_w[2]) * 4

        # update 1/4, 1/2, whole images
        self.im_4 = np.roll(self.im_4, (shift_y - self.shift_h[2], shift_x - self.shift_w[2]), (0, 1))
        self.im_2 = np.roll(self.im_2, (2 * (shift_y - self.shift_h[2]), 2 * (shift_x - self.shift_w[2])), (0, 1))
        self.im = np.roll(self.im, (4 * (shift_y - self.shift_h[2]), 4 * (shift_x - self.shift_w[2])), (0, 1))
        print('1/4 image is aligned.')

    def py_align_2(self, target_im):
        '''
        Function: align 1/2 iamge and update 1/2, whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align 1/2 image
        # using roberts filter to get edge information
        shift_y, shift_x = align(roberts(self.im_2), roberts(self.target.im_2), self.shift_rate[1], 0.35)
        self.info['h'] += (shift_y - self.shift_h[1]) * 2
        self.info['w'] += (shift_x - self.shift_w[2]) * 2

        # update 1/2, whole images
        self.im_2 = np.roll(self.im_2, (shift_y - self.shift_h[1], shift_x - self.shift_w[2]), (0, 1))
        self.im = np.roll(self.im, (2 * (shift_y - self.shift_h[1]), 2 * (shift_x - self.shift_w[1])), (0, 1))
        print('1/2 image is aligned.')
    
    def py_align(self, target_im):
        '''
        Function: align whole iamge and whole images
        target_im: target image array
        '''
        self.target = target(target_im)

        # align whole image
        # using roberts filter to get edge information
        shift_y, shift_x = align(roberts(self.im), roberts(self.target.im), self.shift_rate[0], 0.35)
        self.info['h'] += (shift_y - self.shift_h[0])
        self.info['w'] += (shift_x - self.shift_w[0])

        # update whole image
        self.im = np.roll(self.im, (shift_y - self.shift_h[0], shift_x - self.shift_w[0]), (0, 1))
        print('whole image is aligned.')
    
    def save_info(self, file_name):
        self.info['h'] = int(self.info['h'])
        self.info['w'] = int(self.info['w'])
        with open(file_name, 'w') as f:
            json.dump(self.info, f)