import numpy as np 
from morph import morph, change
import skimage.io as io
import skimage
from os.path import join, exists
from utils import mkdir
from triangles import ge_tri
import json
import argparse
import os
from point import co_points
from gif import make_gif
from dataset import ge_detaset_info
from average import ge_average
from scipy.spatial import Delaunay
from single_point import point_image
from os.path import split, splitext

 
def parse_args():
    parser = argparse.ArgumentParser(description='P3 code')
    parser.add_argument('cmd', choices=['define_correspondences', 'mid_face', 'morph_sequence', 'mean_face', 'caricature', 'change', 'point_image', \
                            'change_geo'], 
                        help=" choose from 'define_correspondences', 'mid_face', 'morph_sequence', 'change', 'point_image', 'change_geo', 'mean_face' \
                            and 'caricature'")
    parser.add_argument('-d', '--dir', default='output', help='output dir')
    parser.add_argument('-a', '--img_a', default='', help='source image')
    parser.add_argument('-b', '--img_b', default='', help='destination image(when creating caricature, image b should be average image)')
    parser.add_argument('-r', '--trans_rate', default=0.5, type=float, help='transition rate(for caricature, trans_rate should be negtive)')
    parser.add_argument('--pt_a', default='', help='points_a file path')
    parser.add_argument('--pt_b', default='', help='points_b file path')
    parser.add_argument('--dataset_info', default='', help='the json file of dataset info')
    parser.add_argument('--dataset', default='', help='the path to Danes dataset')
    parser.add_argument('-c', '--img_c', default='', help='image to be changed')
    parser.add_argument('--pt_c', default='', help='points_c file path')
    parser.add_argument('-f', '--feature', default='1m', help='the feature to use in the dataset')
    # parser.add_argument('--change_geo', default=False, type=bool, help='change the image geometry(only use when cmd==mean_face)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mkdir(args.dir)
    if args.cmd=='define_correspondences':
        assert args.img_a!='' and args.img_b!='', 'Defining correspondences need two input images: img_a & img_b!'
        sub_path_corr = join(args.dir, 'point_correspondences')
        mkdir(sub_path_corr)
        co_points(args.img_a, args.img_b, sub_path_corr)
        print('Defining correspondences finished')
    
    elif args.cmd=='point_image':
        assert args.img_a!='', 'Point image need one input images: img_a!'
        sub_path_point = join(args.dir, 'points')
        mkdir(sub_path_point)
        point_image(args.img_a, sub_path_point)

    elif args.cmd=='mid_face' or args.cmd=='morph_sequence' or args.cmd=='caricature' or args.cmd=='change_geo':
        assert args.img_a!='' and args.img_b!='', \
        'Creating mid-way faces, morph sequence or caricatures need two input images: img_a & img_b!'

        img_a = io.imread(args.img_a)
        img_b = io.imread(args.img_b)
        pts_path_a = args.pt_a
        pts_path_b = args.pt_b
        if args.pt_a=='' or args.pt_b=='':
            print('Defining correspondences')
            sub_path_pts = join(args.dir, 'point_correspondences')
            mkdir(sub_path_pts)
            co_points(args.img_a, args.img_b, sub_path_pts)
            pts_path_a = join(sub_path_pts, 'points_a.json')
            pts_path_b = join(sub_path_pts, 'points_b.json')
            print('Defining correspondences finished')
        tri = ge_tri(pts_path_a, pts_path_b)
        with open(pts_path_a) as f:
            points_a = np.array(json.load(f))
        with open(pts_path_b) as f:
            points_b = np.array(json.load(f))

        if args.cmd=='mid_face':
            sub_path_midface = join(args.dir, 'mid_way_face')
            mkdir(sub_path_midface)
            mid_way_img = morph(img_a, img_b, points_a, points_b, tri, 0.5, 0.5)
            io.imsave(join(sub_path_midface, 'mid_way_face.jpg'), mid_way_img)
            print('Creating mid-way face finished')
        
        elif args.cmd=='morph_sequence':
            sub_path_sequence = join(args.dir, 'morph_sequence')
            mkdir(sub_path_sequence)
            for i in np.arange(0, 46):
                frac = float(i) / 45.
                sequence_image = morph(img_a, img_b, points_a, points_b, tri, frac, frac)
                io.imsave(join(sub_path_sequence, str(i) + '.jpg'), sequence_image)
            
            make_gif(sub_path_sequence, 'morph_sequence.gif')
            print('Creating morph sequence face finished')
        
        elif args.cmd=='caricature':
            if args.trans_rate>0:
                caricature_rate = -0.5
            else:
                caricature_rate = args.trans_rate

            sub_path_caricature = join(args.dir, 'caricatures')
            mkdir(sub_path_caricature)
            caricature = morph(img_a, img_b, points_a, points_b, tri, caricature_rate, 0)
            io.imsave(join(sub_path_caricature, 'caricature.jpg'), caricature)
            print('Creating caricature face finished')
        
        elif args.cmd=='change_geo':
            sub_path_change_geo = join(args.dir, 'change_geo')
            mkdir(sub_path_change_geo)
            changed_image = morph(img_a, img_b, points_a, points_b, tri, 0, 0, True)
            image_fn = splitext(split(args.img_a)[-1])[0] + '_' + splitext(split(args.img_b)[-1])[0] + '.jpg'
            io.imsave(join(sub_path_change_geo, image_fn), changed_image)
        
    elif args.cmd=='mean_face':
        assert args.dataset!='', "Creating mean face need a dataset path!"
        dataset_info_f = args.dataset_info
        if args.dataset_info=='':
            print("Creating dataset info")
            sub_path_datasetinfo = join(args.dir, 'dataset_info')
            mkdir(sub_path_datasetinfo)
            ge_detaset_info(args.dataset, sub_path_datasetinfo, 'dataset_info.json', args.feature)
            dataset_info_f = join(sub_path_datasetinfo, args.feature + '_'+ 'dataset_info.json')
            print("Creating dataset info finished")
        
        sub_path_average = join(args.dir, 'average')
        mkdir(sub_path_average)
        with open(dataset_info_f) as f:
            dataset_info = json.load(f)
        
        ge_average(args.dataset, dataset_info, sub_path_average, args.feature + '_'+ 'average.jpg')
        print("Creating average image finished")
    
    elif args.cmd=='change':
        assert args.img_a!='' and args.img_b!='' and args.img_c!='c', \
            'change need 3 images'
        assert args.pt_a!='' and args.pt_b!='' and args.pt_c!='', \
            'change need 3 points list'
        
        sub_path_change = join(args.dir, 'change')
        mkdir(sub_path_change)
        
        img_a = io.imread(args.img_a)
        img_b = io.imread(args.img_b)
        img_c = io.imread(args.img_c)

        with open(args.pt_a) as f:
            points_a = np.array(json.load(f))
        with open(args.pt_b) as f:
            points_b = np.array(json.load(f))
        with open(args.pt_c) as f:
            points_c = np.array(json.load(f))
        
        tri = Delaunay((points_a + points_b + points_c) / 3)
        
        change_img = change(img_c, img_a, img_b, points_c, points_a, points_b, tri)
        io.imsave(join(sub_path_change, 'change.jpg'), change_img)


        

        
        








