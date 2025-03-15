#!.test-env/bin/python3
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

from time import time
from xray_angio_3d import reconstruction
from metrics import *
from util import *
from vessel_tree_generator.module import *

OUTPUT_DIR="data/"
OUTPUT_FILENAME="experiment_results.csv"

def reconstruct_and_measure(gt, xinfs):
    start = time()
    reconstructed = reconstruction(xinfs, True)
    elapsed_s = time() - start
    hat = np.array(reconstructed['vessel'])
    gt_vox = voxelize_points(gt)
    hat_vox = voxelize_points(hat)

    sum_dice2d = 0 # will be averaged out
    sum_iou2d = 0

    if 'iou2d' in METRICS or 'dice2d' in METRICS:
        for xinfo in xinfs:
            spacing = xinfo.acquisition_params['spacing_r']
            img_hat = make_projection(gt, 
                xinfo.acquisition_params['alpha'], xinfo.acquisition_params['beta'], 
                xinfo.acquisition_params['sod'], xinfo.acquisition_params['sid'],
                (spacing, spacing), 512
            )
            img_gt = xinfo.image
            sum_dice2d += dice2d(img_gt, img_hat)
            sum_iou2d += iou2d(img_gt, img_hat)


    return {
        "Time [s]" : elapsed_s if 'time' in METRICS else None,
        "Dice 3D [%]" : dice3d(gt_vox, hat_vox) * 100 if 'dice' in METRICS else None,
        "IoU 3D [%]" : iou3d(gt_vox, hat_vox) * 100 if 'iou' in METRICS else None,
        "Chamfer distance 3D [mm]" : chamfer3d(gt_vox, hat_vox) * 1000 if 'chamfer' in METRICS else None,
        'Avg IoU 2D [%]' : sum_iou2d / len(xinfs) if 'iou2d' in METRICS else None,
        'Avg Dice 2D [%]' : sum_dice2d / len(xinfs) if 'dice2d' in METRICS else None,
    }

def case_general(config):
    df = []
    CUTOFF=10
    number_of_vessels = min(config['experiment']['dataset']['num_vessels_for_each'], CUTOFF)

    for tortosity in config['experiment']['dataset']['tortosity']:
        for ix in range(0, number_of_vessels):
            vessel = f"{tortosity}:{ix}"
            print(vessel)
            gt, xinfs, _ = load_sample(NAME, ix, tortosity)
            res = reconstruct_and_measure(gt, xinfs)
            res = pd.DataFrame([{
                    "vessel" : vessel,
                    "tortosity" : tortosity,
                    **res
            }])
            df.append(res)    
    result = pd.concat(df, ignore_index=True)
    result.to_csv(os.path.join(OUTPUT_DIR, NAME, OUTPUT_FILENAME))

def case_projections(config):
    df = []
    gt, xinfs, _ = load_sample(NAME, 0, "moderate")

    for no_projections in range(1, config):
        print(no_projections)
        for _ in range(config['experiment']['testcase']['repeat_selection']):
            res = reconstruct_and_measure(gt, np.random.choice(xinfs, no_projections, replace=False))
            res = pd.DataFrame([{
                    "# Projections": no_projections,
                    **res
            }])
            df.append(res)    
    result = pd.concat(df, ignore_index=True)
    result.to_csv(os.path.join(OUTPUT_DIR, NAME, OUTPUT_FILENAME))

def case_angles(config):
    df = []

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    gt, xinfs, _ = load_sample(NAME, 0, "moderate")
    triplet_count = config["experiment"]["testcase"]["triplet_count"]
    random.seed(43)
    random.shuffle(xinfs)
    random_triplets = [random.sample(xinfs, 3) for _ in range(triplet_count)]
    for i, triplet in enumerate(random_triplets):
        print(f"{i + 1} / {triplet_count}")
        res = reconstruct_and_measure(gt, triplet)
        vectors = [
            [triplet[0].acquisition_params['alpha'], triplet[0].acquisition_params['beta']],
            [triplet[1].acquisition_params['alpha'], triplet[1].acquisition_params['beta']],
            [triplet[2].acquisition_params['alpha'], triplet[2].acquisition_params['beta']]
        ]
        vectors = [normalize(v) for v in vectors]
        matrix = np.array(vectors)
        _, s, _ = np.linalg.svd(matrix)
        s = min(s)
        
        res = pd.DataFrame([{
            "v0_x": vectors[0][0],
            "v0_y": vectors[0][1],
            "v1_x": vectors[1][0],
            "v1_y": vectors[1][1],
            "v2_x": vectors[2][0],
            "v2_y": vectors[2][1], 
            "Singular_2": s,     # The smallest singular value gives a measure of linear dependence.
            **res,
        }])
        df.append(res)   
    result = pd.concat(df, ignore_index=True)
    result.to_csv(os.path.join(OUTPUT_DIR, NAME, OUTPUT_FILENAME))

def case_noise(config):
    df = []
    gt, xinfs, noise = load_sample(NAME, 0, "moderate")
    picked_noise = noise[0]
    
    # the last two don't have noise
    xinfo1 = xinfs[-1]
    xinfo2 = xinfs[-2]

    for i, (t_x, t_y) in enumerate(picked_noise):
        print(f"{i}/{len(picked_noise)}")
        xinfo0 = xinfs[i]
        res = reconstruct_and_measure(gt, [xinfo0, xinfo1, xinfo2])
        res = pd.DataFrame([{
            'translation' : [t_x, t_y],
            'translation_norm' : np.linalg.norm([t_x, t_y]),
            'scale' : 1,
            **res
        }])
        df.append(res)
    result = pd.concat(df, ignore_index=True)
    print(result)
    result.to_csv(os.path.join(OUTPUT_DIR, NAME, OUTPUT_FILENAME))


def case_scaling(config):
    pass

if __name__ == "__main__":
    global config, METRICS, NAME
    config = load_config()
    NAME = config['experiment']['name']
    METRICS = config['experiment']['testcase']['metrics']
    TYPE = config['experiment']['testcase']['type']

    if  TYPE == 'projections':
        case_projections(config)
    elif TYPE == 'angles':
        case_angles(config)
    elif TYPE == 'general':
        case_general(config)
    elif TYPE == 'noise':
        case_noise(config)
    else:
        raise Exception("Invalid experiment type")
