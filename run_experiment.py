#!.test-env/bin/python3
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from time import time
from xray_angio_3d import reconstruction
from metrics import *
from util import *

OUTPUT_DIR="data/"
OUTPUT_FILENAME="experiment_results.csv"

def reconstruct_and_measure(gt, xinfs):
    start = time()
    reconstructed = reconstruction(xinfs, True)
    elapsed_s = time() - start
    hat = np.array(reconstructed['vessel'])
    gt_vox = voxelize_points(gt)
    hat_vox = voxelize_points(hat)

    return {
        "Time [s]" : elapsed_s if 'time' in METRICS else None,
        "Dice 3D [%]" : dice3d(gt_vox, hat_vox) * 100 if 'dice' in METRICS else None,
        "IoU 3D [%]" : iou3d(gt_vox, hat_vox) * 100 if 'iou' in METRICS else None,
        "Chamfer distance 3D [mm]" : chamfer3d(gt_vox, hat_vox) * 1000 if 'chamfer' in METRICS else None,
    }

def case_projections(config):
    df = []
    gt, xinfs = load_sample(NAME, 0, "moderate")

    for no_projections in range(1, 16):
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

    gt, xinfs = load_sample(NAME, 0, "moderate")
    triplets  = list(itertools.combinations(xinfs, 3))
    for i, triplet in enumerate(triplets):
        print(f"{i}/{len(triplets)}")
        res = reconstruct_and_measure(gt, triplet)
        res = pd.DataFrame([{
            "Alpha 0": triplet[0].acquisition_params['alpha'],
            'Beta 0': triplet[0].acquisition_params['beta'],
            "Alpha 1": triplet[1].acquisition_params['alpha'],
            'Beta 1': triplet[1].acquisition_params['beta'],
            "Alpha 2": triplet[2].acquisition_params['alpha'],
            'Beta 2': triplet[2].acquisition_params['beta'],
            **res,
        }])
        df.append(res)   
    result = pd.concat(df, ignore_index=True)
    print(result)

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

