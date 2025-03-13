#!.test-env/bin/python3
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
from xray_angio_3d import reconstruction
from metrics import *
from util import *

def reconstruct_and_measure(gt, xinfs):
    start = time()
    reconstructed = reconstruction(xinfs)
    hat = np.array(reconstructed['vessel'])
    gt_vox = voxelize_points(gt)
    hat_vox = voxelize_points(hat)
    elapsed_s = time() - start

    return {
        "Time [s]" : elapsed_s,
        "Dice (3D)" : dice3d(gt_vox, hat_vox),
        "Chamfer distance [mm]" : chamfer3d(gt_vox, hat_vox) * 1000,
        "clDice3D" : cl_dice3d(gt_vox, hat_vox)
    }

if __name__ == "__main__":
    result = reconstruct_and_measure(*load_sample(0, "tortuous"))
    print(result)
    pass
