#!.test-env/bin/python3
# script for synthetic dataset generation
# generated sample naming scheme is:
# <OUTPUT_DIR>/<experiment_name>/<tortosity>/<ix>.gt
# <OUTPUT_DIR>/<experiment_name>/<tortosity>/<ix>.xinfs

import numpy as np
import matplotlib.pyplot as plt
import yaml
import itertools
import argparse
import pickle
import os
import random

from xray_angio_3d import reconstruction, XRayInfo
from vessel_tree_generator.module import *
from util import load_config

OUTPUT_DIR="data/"
VESSEL_TYPE="RCA"
IMG_DIM=512

def parse_config(config):
    global NAME
    global SID, SOD, SPACING, COUNT, TORTOSITY
    global RANDOM_ANGLES, GRID_ANGLES, ANGLE_PAIRS

    print("loading config...")
  
    NAME = config['experiment']['name']
    SID=float(config['experiment']['dataset']['SID'])
    SOD=float(config['experiment']['dataset']['SOD'])
    SPACING=float(config['experiment']['dataset']['SPACING'])
    TORTOSITY = config['experiment']['dataset']['tortosity']
    COUNT=int(config['experiment']['dataset']['num_vessels_for_each'])
    RANDOM_ANGLES = config['experiment']['dataset'].get('random_angles')
    GRID_ANGLES = config['experiment']['dataset'].get('grid_angles')
    ANGLE_PAIRS = config['experiment']['dataset'].get('angle_pairs')

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
    if RANDOM_ANGLES is not None:
        pc = RANDOM_ANGLES['pair_count']
        min = RANDOM_ANGLES['min_angle']
        max = RANDOM_ANGLES['max_angle']
        pairs = []
        for _ in range(pc):
            alpha = random.randint(min, max)
            beta = random.randint(min, max)
            pairs.append((alpha, beta))
        return pairs
    if GRID_ANGLES is not None:
        min = GRID_ANGLES['min_angle']
        max = GRID_ANGLES['max_angle']
        step = GRID_ANGLES['step']
        alpha_values = np.arange(min, max + 1, step)
        beta_values = np.arange(0, max + 1, step) 
        pairs = [(alpha, beta) for alpha in alpha_values for beta in beta_values]
        return pairs
    if ANGLE_PAIRS is not None:
        return ANGLE_PAIRS
    raise Exception("No angle generation mode specified")

def mkdirs():
    print(f"making necessary dirs...")
    os.makedirs(os.path.join(OUTPUT_DIR, NAME), exist_ok=True)
    for tortosity in TORTOSITY:
        tortosity_dir = os.path.join(OUTPUT_DIR, NAME, str(tortosity))
        os.makedirs(tortosity_dir, exist_ok=True)

def save_data(ix, tortosity, gt, xinfs):
    print(f'saving     {tortosity}:{ix}')
    gt_filename = os.path.join(OUTPUT_DIR, NAME, tortosity, f'{ix}.gt')
    xinfs_filename = os.path.join(OUTPUT_DIR, NAME, tortosity, f'{ix}.xinfs')
    with open(gt_filename, 'wb') as f:
        pickle.dump(gt, f)
    with open(xinfs_filename, 'wb') as f:
        pickle.dump(xinfs, f)

if __name__ == "__main__":
    parse_config(load_config())
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
