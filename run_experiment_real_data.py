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
VISUALIZE_INTERMEDIATE=False

def isocenter_motion_correction(img, angle, translation):
    height, width = img.shape[:2]
    center = (height/2, width/2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    M[0,2] = translation[0]
    M[1,2] = translation[1]
    return cv2.warpAffine(img, M, (width, height))

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

    xinfo3 = XRayInfo()
    xinfo3.image = cv2.imread(os.path.join(DATA_DIR, "ex23_fr11.png"), cv2.IMREAD_GRAYSCALE)
    xinfo3.width = 512
    xinfo3.height = 512
    xinfo3.acquisition_params = {
        'sid': 1.113,
        'sod': 0.742,
        'alpha': -29.7,
        'beta': 1.5,
        'spacing_r': 0.40425 / 1000,
        "spacing_c": 0.40425 / 1000
    }

    experiments = [
        ([xinfo0, xinfo1, xinfo2], xinfo3),
        ([xinfo0, xinfo1, xinfo3], xinfo2),
        ([xinfo0, xinfo2, xinfo3], xinfo1),
        ([xinfo1, xinfo2, xinfo3], xinfo0)      
    ]

    # motion correction
    xinfo2.image = isocenter_motion_correction(xinfo2.image, -10, [40, -30])
    xinfo3.image = isocenter_motion_correction(xinfo3.image, -8, [90, 0])

    pts = []
    for i, (xinfs, test) in enumerate(experiments):
        reconstructed = reconstruction(xinfs, True)
        hat = reconstructed['vessel']
        pts.append(hat)
        X_hat, Y_hat, Z_hat = zip(*hat)
        sum_dice2d = 0
        sum_dice2d_eroded = 0
        for i, xinfo in enumerate(xinfs + [test]):
            print(xinfo.acquisition_params)
            spacing = xinfo.acquisition_params['spacing_r']
            
            img_hat = make_projection(np.array(hat), 
                xinfo.acquisition_params['alpha'], xinfo.acquisition_params['beta'], 
                xinfo.acquisition_params['sod'], xinfo.acquisition_params['sid'],
                (spacing, spacing), 512
            )
            img_gt = xinfo.image
            dilation_radius = int(np.ceil(0.002 / spacing))
            dilation_radius_gt = int(np.ceil(0.002 / spacing))
            img_hat = ndimage.binary_dilation(img_hat, structure=np.ones((dilation_radius, dilation_radius)))
            img_gt_eroded = ndimage.binary_dilation(img_gt, structure=np.ones((dilation_radius_gt, dilation_radius_gt)))
            
            if VISUALIZE_INTERMEDIATE:
                plt.imshow(np.stack(((img_hat * 255).astype(np.uint8), img_gt, np.zeros_like(img_hat)), axis=0).T)
                plt.show()

            sum_dice2d += dice2d(img_gt, img_hat)
            sum_dice2d_eroded += dice2d(img_gt_eroded, img_hat)
        print(f"Average dice2d {sum_dice2d / len(xinfs)}")
        print(f"Average eroded dice2d {sum_dice2d_eroded / len(xinfs)}")

    fig = plt.figure(figsize=(15, 5))
    elevations = [30, 0, 60]
    azim_angles = [45, 0, 135]
    
    for i, (elev, azim) in enumerate(zip(elevations, azim_angles)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        for j, hat in enumerate(pts):
            X_hat, Y_hat, Z_hat = zip(*hat)
            ax.scatter(X_hat, Y_hat, Z_hat, label=f"subset {j}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))
    plt.tight_layout()
    plt.savefig("notebooks/figures/qual_eval.png")
    plt.show()
