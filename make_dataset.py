# -*- coding: <encoding name> -*-
"""
data processing.
"""
from __future__ import print_function, division
import h5py
import scipy.io as io
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.spatial
import os
# %matplotlib inline


################################################################################
# generate density maps of the ShanghaiTech dataset.
# the density maps are stored in the density directory,
# which is at the same level as the images directory.
################################################################################
def generate_density():
    # root directory
    root = 'datasets/ShanghaiTech_Dataset'
    # PartA train and test set directory
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    # # PartB train and test set directory
    # part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    # part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

    path_sets = [part_A_train, part_A_test]

    # density map storage directory
    for path in path_sets:
        dir_path, _ = os.path.split(path)
        path = os.path.join(dir_path, 'density')
        if not os.path.exists(path):
            os.makedirs(path)

    # acquire image path
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    # generate density map
    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'density'), 'w') as hf:
            hf['density'] = k

    print("Over!!!")


################################################################################
# internal function.
################################################################################
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype = np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    # generate density map
    generate_density()

    pass