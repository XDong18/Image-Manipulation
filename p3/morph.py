from triangles import ge_tri
from affine import computeAffine
from skimage.draw import polygon
import numpy as np 

def bound(target_img, h, w):
    target_img_0 = target_img[1]
    target_img_1 = target_img[0]

    target_img_0[np.where(target_img_0>h-1)] = h-1
    target_img_0[np.where(target_img_0<0)] = 0
    target_img_1[np.where(target_img_1>w-1)] = w-1
    target_img_1[np.where(target_img_1<0)] = 0
    return (target_img_0, target_img_1)


def morph(img_a, img_b, points_a, points_b, tri, warp_frac, dissolve_frac, change_geo=False):
    if change_geo==True:
        points_ave = points_b
    else:
        points_ave = (1-warp_frac) * points_a + warp_frac * points_b

    tri_c = tri
    h, w, _ = img_a.shape
    mid_img = np.zeros(img_a.shape, dtype=np.float64)
    #print(points_ave[tri_c.simplices[0]])
    for tri_idx in tri_c.simplices:
        # print(tri_idx)
        # print(points_a[tri_idx])
        re_affine_a = computeAffine(points_ave[tri_idx], points_a[tri_idx])
        re_affine_b = computeAffine(points_ave[tri_idx], points_b[tri_idx])
        tri_pts = points_ave[tri_idx]
        r = tri_pts[:,1]
        c = tri_pts[:,0]
        rr, cc = polygon(r, c)
        ones = np.ones(rr.size, dtype=np.int64)
        mask_pts = np.stack((cc, rr, ones), axis = 0)
        target_a = np.dot(re_affine_a, mask_pts).astype(np.int64)
        target_b = np.dot(re_affine_b, mask_pts).astype(np.int64)
 
        a_0, a_1 = bound(target_a, h, w)
        b_0, b_1 = bound(target_b, h, w)
        # target_a[np.where(target_a>749)] = 749
        # target_b[np.where(target_b>749)] = 749
        

        tri_img_a = img_a[a_0, a_1]
        tri_img_b = img_b[b_0, b_1]
        if change_geo==True:
            mid_img[rr, cc] =  tri_img_a
        else:
            mid_img[rr, cc] =  tri_img_a * (1-dissolve_frac) + tri_img_b * dissolve_frac

    return mid_img


def average(img_list, point_list, target_points, tri):
    num_img = len(img_list)
    ave_img = np.zeros(img_list[0].shape, dtype=np.float64)
    h, w, _ = img_list[0].shape

    for idx, img in enumerate(img_list):
        pts = point_list[idx]
        for tri_idx in tri.simplices:
            # print(target_points, pts)
            re_affine = computeAffine(target_points[tri_idx], pts[tri_idx])
            tri_pts = target_points[tri_idx]
            r = tri_pts[:,1]
            c = tri_pts[:,0]
            rr, cc = polygon(r, c)
            ones = np.ones(rr.size, dtype=np.int64)
            mask_pts = np.stack((cc, rr, ones), axis = 0)
            target_img = np.dot(re_affine, mask_pts).astype(np.int64)
            target_img_0 = target_img[1]
            target_img_1 = target_img[0]

            target_img_0[np.where(target_img_0>h-1)] = h-1
            target_img_0[np.where(target_img_0<0)] = 0
            target_img_1[np.where(target_img_1>w-1)] = w-1
            target_img_1[np.where(target_img_1<0)] = 0

            tri_img = img[target_img_0, target_img_1]
            ave_img[rr, cc] =  ave_img[rr, cc] + tri_img * (1/num_img)
    
    return ave_img

def change(img, img_a, img_b, points, points_a, points_b, tri):
    
    points_ave = points + (points_b - points_a)
    print((points_b - points_a))
    tri_c = tri
    h, w, _ = img_a.shape
    mid_img = np.zeros(img_a.shape, dtype=np.float64)
    for tri_idx in tri_c.simplices:
        re_affine_a = computeAffine(points_ave[tri_idx], points_a[tri_idx])
        re_affine_b = computeAffine(points_ave[tri_idx], points_b[tri_idx])
        re_affine = computeAffine(points_ave[tri_idx], points[tri_idx])

        tri_pts = points_ave[tri_idx]
        r = tri_pts[:,1]
        c = tri_pts[:,0]
        rr, cc = polygon(r, c)
        ones = np.ones(rr.size, dtype=np.int64)
        mask_pts = np.stack((cc, rr, ones), axis = 0)

        target_a = np.dot(re_affine_a, mask_pts).astype(np.int64)
        target_b = np.dot(re_affine_b, mask_pts).astype(np.int64)
        target = np.dot(re_affine, mask_pts).astype(np.int64)
 
        a_0, a_1 = bound(target_a, h, w)
        b_0, b_1 = bound(target_b, h, w)
        x_0, x_1 = bound(target, h, w)

        tri_img_a = img_a[a_0, a_1]
        tri_img_b = img_b[b_0, b_1]
        tri_img = img[x_0, x_1]
        mid_img[rr, cc] =  tri_img * 1 + (tri_img_b - tri_img_a) * 0
    
    return mid_img