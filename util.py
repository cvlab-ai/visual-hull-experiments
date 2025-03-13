import os
import pickle
import numpy as np

INPUT_DIR="data/"
VOXELIZATION_THRESH_DEFAULT=0.003

def voxelize_points(points, voxel_size=VOXELIZATION_THRESH_DEFAULT):
    scaled_points = np.floor(points / voxel_size).astype(int)
    unique_voxels = np.unique(scaled_points, axis=0)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2.0
    return voxel_centers

def load_sample(ix, tortosity):
    try:
        tortosity_dir = os.path.join(INPUT_DIR, str(tortosity))
        gt_filename = os.path.join(tortosity_dir, f'{ix}.gt')
        xinfs_filename = os.path.join(tortosity_dir, f'{ix}.xinfs')
        with open(gt_filename, 'rb') as gt_file:
            gt = pickle.load(gt_file)
        with open(xinfs_filename, 'rb') as xinfs_file:
            xinfs = pickle.load(xinfs_file)
    except:
        gt, xinfs = None, None
    return gt, xinfs