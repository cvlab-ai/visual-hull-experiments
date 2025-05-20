#!.test-env/bin/python3
import os
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from xray_angio_3d import reconstruction, XRayInfo
from metrics import *
from util import *
from vessel_tree_generator.module import *

DATA_DIR = "data/real"

if __name__ == "__main__":
    xinfs = []

    xinfo0 = XRayInfo()
    xinfo0.image = cv2.imread(os.path.join(DATA_DIR, "ex6_fr14.png"), cv2.IMREAD_GRAYSCALE)
    xinfo0.width = 512
    xinfo0.height = 512
    xinfo0.acquisition_params = {
        'sid': 1.027,
        'sod': 0.739,
        'alpha': 22.7,
        'beta': 19.5,
        'spacing_r': 0.40425 / 1000,
        "spacing_c": 0.40425 / 1000
    }

    xinfo1 = XRayInfo()
    xinfo1.image = cv2.imread(os.path.join(DATA_DIR, "ex7_fr14.png"), cv2.IMREAD_GRAYSCALE)
    xinfo1.width = 512
    xinfo1.height = 512
    xinfo1.acquisition_params = {
        'sid': 1.200,
        'sod': 0.749,
        'alpha': -48.2,
        'beta': 25.4,
        'spacing_r': 0.40425 / 1000,
        "spacing_c": 0.40425 / 1000
    }

    xinfo2 = XRayInfo()
    xinfo2.image = cv2.imread(os.path.join(DATA_DIR, "ex8_fr13.png"), cv2.IMREAD_GRAYSCALE)
    xinfo2.width = 512
    xinfo2.height = 512
    xinfo2.acquisition_params = {
        'sid': 1.126,
        'sod': 0.739,
        'alpha': -3.4,
        'beta': -36.3,
        'spacing_r': 0.40425 / 1000,
        "spacing_c": 0.40425 / 1000
    }

    xinfs = [
        xinfo0,
        xinfo2,
        xinfo1,
    ]

    reconstructed = reconstruction(xinfs, True)
    hat = reconstructed['vessel']
    X_hat, Y_hat, Z_hat = zip(*hat)
    sum_dice2d = 0
    for i, xinfo in enumerate(xinfs):
        print(xinfo.acquisition_params)
        spacing = xinfo.acquisition_params['spacing_r']
        
        img_hat = make_projection(np.array(hat), 
            xinfo.acquisition_params['alpha'], xinfo.acquisition_params['beta'], 
            xinfo.acquisition_params['sod'], xinfo.acquisition_params['sid'],
            (spacing, spacing), 512
        )
        img_gt = xinfo.image
        dilation_radius = int(np.ceil(0.003 / spacing))
        dilation_radius_gt = int(np.ceil(0.003 / spacing))
        img_hat = ndimage.binary_dilation(img_hat, structure=np.ones((dilation_radius, dilation_radius)))
        img_gt = ndimage.binary_dilation(img_gt, structure=np.ones((dilation_radius_gt, dilation_radius_gt)))
        plt.imshow(np.stack(((img_hat * 255).astype(np.uint8), (img_gt * 255).astype(np.uint8), np.zeros_like(img_hat)), axis=0).T)
        plt.show()
        sum_dice2d += dice2d(img_gt, img_hat)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_hat, Y_hat, Z_hat)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    print(f"Average dice2d {sum_dice2d / len(xinfs)}")