#!.test-env/bin/python3
# script for synthetic dataset generation
# generated sample naming scheme is:
# <OUTPUT_DIR>/<tortosity>/<ix>.gt
# <OUTPUT_DIR>/<tortosity>/<ix>.xinfs

import numpy as np
import matplotlib.pyplot as plt
import yaml
import itertools
import argparse
import pickle
import os

from xray_angio_3d import reconstruction, XRayInfo
from vessel_tree_generator.module import *
from math import radians

OUTPUT_DIR="data/"
VESSEL_TYPE="RCA"
IMG_DIM=512

def load_config(config_path):
    global SID, SOD, SPACING  
    global TORTOSITY
    global COUNT, ALPHA_RANGE_DEG, BETA_RANGE_DEG

    print("loading config...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    SID=float(config['generate']['SID'])
    SOD=float(config['generate']['SOD'])
    SPACING=float(config['generate']['SPACING'])
    TORTOSITY = config['generate']['tortosity']
    COUNT=int(config['generate']['num_vessels_for_each'])
    ALPHA_RANGE_DEG = config['generate']['alpha_range_deg']
    BETA_RANGE_DEG = config['generate']['beta_range_deg']

def ensure_generate_vessel_3d(tree_path):
    # vessel can be None due to invalid subsampling
    rng = np.random.default_rng()
    vessel = None
    while vessel is None: vessel, _, _ =  generate_vessel_3d(rng, VESSEL_TYPE, tree_path, True, False)
    return vessel

def generate_tree(tree_path, angle_pairs):
    gt = ensure_generate_vessel_3d(tree_path)
    xinfs = []

    for (alpha, beta) in angle_pairs:
        projection = make_projection(gt, 
            alpha, beta, 
            SOD, SID, 
            (SPACING, SPACING), IMG_DIM
        )
        xinf = XRayInfo()
        xinf.width = 512
        xinf.height = 512
        xinf.image = projection
        xinf.acquisition_params = {
            'sid': SID,
            'sod': SOD,
            'alpha': alpha,
            'beta': beta,
            'spacing_r': SPACING,
            'spacing_c': SPACING
        }
        xinfs.append(xinf)
    return gt, xinfs

def get_angles():
    return list(itertools.product(
        [radians(alpha) for alpha in range(
            ALPHA_RANGE_DEG[0],
            ALPHA_RANGE_DEG[1],
            ALPHA_RANGE_DEG[2]
        )],
        [radians(beta) for beta in range(
            BETA_RANGE_DEG[0],
            BETA_RANGE_DEG[1],
            BETA_RANGE_DEG[2]
        )],
    ))

def mkdirs():
    print(f"making necessary dirs...")
    for tortosity in TORTOSITY:
        tortosity_dir = os.path.join(OUTPUT_DIR, str(tortosity))
        os.makedirs(tortosity_dir, exist_ok=True)

def save_data(ix, tortosity, gt, xinfs):
    print(f'saving     {tortosity}:{ix}')
    gt_filename = os.path.join(OUTPUT_DIR, tortosity, f'{ix}.gt')
    xinfs_filename = os.path.join(OUTPUT_DIR, tortosity, f'{ix}.xinfs')
    with open(gt_filename, 'wb') as f:
        pickle.dump(gt, f)
    with open(xinfs_filename, 'wb') as f:
        pickle.dump(xinfs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    load_config(args.file)
    mkdirs()
    angle_pairs = get_angles()

    for tortosity in TORTOSITY:
        tree_path = f"./vessel_tree_generator/RCA_branch_control_points/{tortosity}"
      
        for ix in range(COUNT):
            print(f"generating {tortosity}:{ix}")
            save_data(
                ix, tortosity,
                *generate_tree(tree_path, angle_pairs)
            )
