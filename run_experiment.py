#!.test-env/bin/python3
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
from xray_angio_3d import reconstruction

INPUT_DIR="data/"

def reconstruct_and_measure(gt, xinfs):
    start = time()
    reconstructed = reconstruction(xinfs)
    elapsed_s = time() - start

    return {
        "time" : elapsed_s
    }

def load_sample(ix, tortosity):
    try:
        tortosity_dir = os.path.join(INPUT_DIR, str(tortosity))
        gt_filename = os.path.join(tortosity_dir, f'{ix}.gt')
        xinfs_filename = os.path.join(tortosity_dir, f'{ix}.xinfs')
        with open(gt_filename, 'rb') as gt_file:
            gt = pickle.load(gt_file)
        with open(xinfs_filename, 'rb') as xinfs_file:
            xinfs = pickle.load(xinfs_file)
    except:
        gt, xinfs = None, None
    return gt, xinfs

if __name__ == "__main__":
    result = reconstruct_and_measure(*load_sample(0, "tortuous"))
    print(result)
    pass
