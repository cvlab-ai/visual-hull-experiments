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
import cv2
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
    global NOISE_XY_RANGE

    print("loading config...")
    
    # mandatory
    NAME = config['experiment']['name']
    SID=float(config['experiment']['dataset']['SID'])
    SOD=float(config['experiment']['dataset']['SOD'])
    SPACING=float(config['experiment']['dataset']['SPACING'])
    TORTOSITY = config['experiment']['dataset']['tortosity']
    COUNT=int(config['experiment']['dataset']['num_vessels_for_each'])

    # optional
    RANDOM_ANGLES = config['experiment']['dataset'].get('random_angles')
    GRID_ANGLES = config['experiment']['dataset'].get('grid_angles')
    ANGLE_PAIRS = config['experiment']['dataset'].get('angle_pairs')
    TRANSLATION_XY_RANGE = config['experiment']['dataset'].get("translation_xy_range", [0,1,1])
    SCALING_XY_RANGE = config['experiment']['dataset'].get("scaling_xy_range", [1,2,1])
    TRANSLATION_XY_RANGE = range(TRANSLATION_XY_RANGE[0], TRANSLATION_XY_RANGE[1], TRANSLATION_XY_RANGE[2])
    SCALING_XY_RANGE = np.arange(SCALING_XY_RANGE[0], SCALING_XY_RANGE[1], SCALING_XY_RANGE[2])
    NOISE_XY_RANGE = list(itertools.product(TRANSLATION_XY_RANGE, TRANSLATION_XY_RANGE, SCALING_XY_RANGE))


def ensure_generate_vessel_3d(tree_path):
    # vessel can be None due to invalid subsampling
    rng = np.random.default_rng()
    vessel = None
    while vessel is None: vessel, _, _ =  generate_vessel_3d(rng, VESSEL_TYPE, tree_path, True, False)
    return vessel

def generate_tree(tree_path, angle_pairs):
    gt = ensure_generate_vessel_3d(tree_path)
    xinfs = []
    translations = []

    for i, (alpha, beta) in enumerate(angle_pairs):
        projection = make_projection(gt, 
            alpha, beta, 
            SOD, SID, 
            (SPACING, SPACING), IMG_DIM
        )
        translation = []
        if i == 0:
            noise_t_xy_range = NOISE_XY_RANGE
        else: 
            noise_t_xy_range = [(0, 0, 1)]
        for (t_x, t_y, s) in noise_t_xy_range:
            xinf = XRayInfo()
            xinf.width = 512
            xinf.height = 512
            xinf.image = scale(translate(projection, (t_x, t_y)), s)
            xinf.acquisition_params = {
                'sid': SID,
                'sod': SOD,
                'alpha': alpha,
                'beta': beta,
                'spacing_r': SPACING,
                'spacing_c': SPACING,
            }
            xinfs.append(xinf)
            translation.append((t_x, t_y, s))
        translations.append(translation)
    return gt, xinfs, translations

def translate(img, vec):
    dx, dy = vec
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(img.astype(np.uint8), M, (w, h)).astype(np.bool)
    return warped

def scale(img, coeff):
    coeff_x, coeff_y = coeff, coeff
    h, w = img.shape[:2]
    center = (w / 2, h / 2)    
    M = np.float32([
        [coeff_x, 0, center[0] - coeff_x * center[0]],
        [0, coeff_y, center[1] - coeff_y * center[1]]
    ])
    warped = cv2.warpAffine(img.astype(np.uint8), M, (w, h))
    return warped.astype(np.bool)

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

def save_data(ix, tortosity, gt, xinfs, translations):
    print(f'saving     {tortosity}:{ix}')
    outdir_filename = os.path.join(OUTPUT_DIR, NAME, tortosity)
    gt_filename = os.path.join(outdir_filename, f'{ix}.gt')
    xinfs_filename = os.path.join(outdir_filename, f'{ix}.xinfs')
    noise_filename = os.path.join(outdir_filename, f'{ix}.noise')
    with open(gt_filename, 'wb') as f:
        pickle.dump(gt, f)
    with open(xinfs_filename, 'wb') as f:
        pickle.dump(xinfs, f)
    with open(noise_filename, 'wb') as f:
        pickle.dump(translations, f)

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
