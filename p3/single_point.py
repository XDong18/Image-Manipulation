import skimage as ski
import skimage.io as io
from skimage.transform import resize
import numpy as np 
import os
import matplotlib.pyplot as plt
import json
from utils import mkdir
from os.path import split, splitext


def point_image(image_a_name, path):
    save_a=splitext(split(image_a_name)[-1])[0] + '.json'
    img_a = io.imread(image_a_name)
    mkdir(path)

    plt.imshow(img_a)
    point_list_a = plt.ginput(n=-1, timeout=0)

    h, w, _ = img_a.shape
    corners = [(0, h-1), (w-1, h-1), (0, 0), (w-1, 0)]
    point_list_a = point_list_a + corners
    with open(os.path.join(path, save_a), 'w') as f:
        json.dump(point_list_a, f)
