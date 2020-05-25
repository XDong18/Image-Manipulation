import skimage as ski
import skimage.io as io
from skimage.transform import resize
import numpy as np 
import os
import matplotlib.pyplot as plt
import json
from utils import mkdir


def co_points(image_a_name, image_b_name, path='information', save_a='points_a.json', save_b='points_b.json'):
    img_b = io.imread(image_b_name)
    img_a = io.imread(image_a_name)
    h, w, _ = img_b.shape

    mkdir(path)

    h, w, _ = img_a.shape
    corners = [(0, h-1), (w-1, h-1), (0, 0), (w-1, 0)]
    plt.imshow(img_a)
    point_list_a = plt.ginput(n=-1, timeout=0)
    with open(os.path.join(path, save_a), 'w') as f:
        json.dump(point_list_a + corners, f)

    plt.imshow(img_b)
    point_list_b = plt.ginput(n=-1, timeout=0)

    with open(os.path.join(path, save_b), 'w') as f:
        json.dump(point_list_b + corners, f)


if __name__ == "__main__":
    co_points("./me_602.jpg", "./george_small.jpg")
    




