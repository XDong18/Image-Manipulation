import numpy as np 


def computeAffine(tri1_pts,tri2_pts):
    # print(tri1_pts)
    mat_tri1 = np.array(
                [[tri1_pts[0][0], tri1_pts[0][1], 1], 
                [tri1_pts[1][0], tri1_pts[1][1], 1], 
                [tri1_pts[2][0], tri1_pts[2][1], 1]])
    mat_tri2_x = tri2_pts[:, 0]
    mat_tri2_y = tri2_pts[:, 1]
    abc_ =  np.linalg.solve(mat_tri1, mat_tri2_x)
    def_ =  np.linalg.solve(mat_tri1, mat_tri2_y)
    affine = np.array([abc_, def_, [0, 0, 1]])

    return affine



