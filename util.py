import os
import pickle
import numpy as np
import argparse
import yaml

INPUT_DIR="data/"
VOXELIZATION_THRESH_DEFAULT=0.003

def voxelize_points(points, voxel_size=VOXELIZATION_THRESH_DEFAULT):
    scaled_points = np.floor(points / voxel_size).astype(int)
    unique_voxels = np.unique(scaled_points, axis=0)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2.0
    return voxel_centers

def load_sample(experiment_name, ix, tortosity):
    try:
        tortosity_dir = os.path.join(INPUT_DIR, experiment_name, str(tortosity))
        gt_filename = os.path.join(tortosity_dir, f'{ix}.gt')
        xinfs_filename = os.path.join(tortosity_dir, f'{ix}.xinfs')
        with open(gt_filename, 'rb') as gt_file:
            gt = pickle.load(gt_file)
        with open(xinfs_filename, 'rb') as xinfs_file:
            xinfs = pickle.load(xinfs_file)
    except:
        gt, xinfs = None, None
    return gt, xinfs

def load_config():
    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.file, "r") as file:
        config = yaml.safe_load(file)
    return config
