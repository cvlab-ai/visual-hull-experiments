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
    # for the computation to take less time we 
    elapsed_s = time() - start

    return {
        "time" : elapsed_s,
        "dice" : dice3d(gt_vox, hat_vox),
    }

if __name__ == "__main__":
    result = reconstruct_and_measure(*load_sample(0, "tortuous"))
    print(result)
    pass
