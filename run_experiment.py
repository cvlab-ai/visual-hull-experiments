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
    reconstructed = reconstruction(xinfs)
    elapsed_s = time() - start
    hat = np.array(reconstructed['vessel'])
    gt_vox = voxelize_points(gt)
    hat_vox = voxelize_points(hat)

    return {
        "Time [s]" : elapsed_s,
        "Dice 3D [%]" : dice3d(gt_vox, hat_vox) * 100,
        "IoU 3D [%]" : iou3d(gt_vox, hat_vox) * 100,
        "Chamfer distance 3D [mm]" : chamfer3d(gt_vox, hat_vox) * 1000,
    }

if __name__ == "__main__":
    df = []

    for tortosity in ["moderate", "tortuous"]:
        for no_projections in range(1, 16):
            for ix in range(0, 10):
                gt, xinfs = load_sample(ix, tortosity)
                res = reconstruct_and_measure(gt, xinfs[:no_projections])
                res = pd.DataFrame([{
                        "Vessel": f"{tortosity}{ix}",
                        "Tortosity": tortosity,
                        "# Projections": no_projections,
                        **res
                    }])
                df.append(res)    
    result = pd.concat(df, ignore_index=True)
    result.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME))
