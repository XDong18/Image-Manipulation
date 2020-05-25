import json
from scipy.spatial import Delaunay
from utils import mkdir
import numpy as np 
import os
import matplotlib.pyplot as plt
import skimage.io as io


def ge_tri(p_a, p_b):
    with open(p_a) as f:
        points_a = json.load(f)

    with open(p_b) as f:
        points_b = json.load(f)

    points_ave = 0.5 * (np.array(points_a, dtype=np.int32) + np.array(points_b, dtype=np.int32))
    tri = Delaunay(points_ave)

    return tri

def ge_tri_array(pts):
    tri = Delaunay(pts)
    return tri

def show_tri(img, p_a, p_b):
    with open(p_a) as f:
        points_a = np.array(json.load(f))

    with open(p_b) as f:
        points_b = np.array(json.load(f))
    
    points_ave = 0.5 * (points_a + points_b)
    tri = Delaunay(points_ave)
    img_a = io.imread(img)
    plt.imshow(img_a)
    plt.triplot(points_a[:,0], points_a[:,1], tri.simplices.copy())
    plt.plot(points_a[:,0], points_a[:,1], 'o')
    plt.savefig('tri_b.jpg')
    plt.show()
    tri = Delaunay(points_ave)

if __name__ == '__main__':
    show_tri("george_small.jpg", 'information\\points_b.json', 'information\\points_a.json')





