#!.test-env/bin/python3
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    tortosity = "moderate"

    for no_projections in range(1, 16):
        print(no_projections)
        for _ in range(config['experiment']['testcase']['repeat_selection']):
            gt, xinfs = load_sample(NAME, 0, "moderate")
            res = reconstruct_and_measure(gt, np.random.choice(xinfs, no_projections, replace=False))
            res = pd.DataFrame([{
                    "Vessel": f"{tortosity}{0}",
                    "Tortosity": tortosity,
                    "# Projections": no_projections,
                    **res
            }])
            df.append(res)    
    result = pd.concat(df, ignore_index=True)
    result.to_csv(os.path.join(OUTPUT_DIR, NAME, OUTPUT_FILENAME))

    print(result)


if __name__ == "__main__":
    # args = parse_args()
    # with open(args.file) as f:
    global config, METRICS, NAME
    config = load_config()
    NAME = config['experiment']['name']
    METRICS = config['experiment']['testcase']['metrics']
    if config['experiment']['testcase']['type'] == 'projections':
        case_projections(config)

    
