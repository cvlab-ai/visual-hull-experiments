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

REAL_DATA_COUNT = 10
DATA_DIR = "data/real"
SAMPLES = [
    ("1.3.6.1.4.1.19291.2.1.2.156249211917942617451923139", "0028.png"),
    ("1.3.6.1.4.1.19291.2.1.2.1562492119179426174519822721", "0049.png"),
    ("1.3.6.1.4.1.19291.2.1.2.1562492119179426174520874733", "0036.png")
]
INSPECTION_NAME = "o1"


def load_format_sample(path, sample_name, filename):
    img_path = os.path.join(path, sample_name, "masks", "vessels", filename)
    df = pd.read_csv(os.path.join(path, "parameters.csv"), sep="\t")
    row_df = df[df['name'] == f"\{sample_name}"]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    xinfo = XRayInfo()
    xinfo.height = img.shape[0]
    xinfo.width = img.shape[1]
    xinfo.image = img
    spacing = 0.308 / 1000,
    xinfo.acquisition_params = {
        'sid': row_df[[' sid']].values[0] / 1000,
        'sod': row_df[[' sod']].values[0] / 1000,
        'alpha': row_df[[' alfa']].values[0],
        'beta': row_df[[' beta']].values[0],
        "spacing_r": spacing,
        "spacing_c": spacing
    }
    return xinfo


def manual_calibration(image):

    return image


if __name__ == "__main__":
    xinfs = []

    for sample_name, file_name in SAMPLES:
        xinfo = load_format_sample(os.path.join(DATA_DIR, "o1"), sample_name, file_name)
        xinfs.append(xinfo)
    xinfs[2].image = manual_calibration(xinfs[2].image)
    reconstructed = reconstruction(xinfs, True)
    hat = reconstructed['vessel']
    X_hat, Y_hat, Z_hat = zip(*hat)
    sum_dice2d = 0
    for xinfo in xinfs:
        print(xinfo.acquisition_params)
        spacing = xinfo.acquisition_params['spacing_r'][0]
        img_hat = make_projection(np.array(hat), 
            xinfo.acquisition_params['alpha'], xinfo.acquisition_params['beta'], 
            xinfo.acquisition_params['sod'], xinfo.acquisition_params['sid'],
            (spacing, spacing), 512
        )
        img_gt = xinfo.image
        dilation_radius = int(np.ceil(VOXELIZATION_THRESH_DEFAULT / spacing))
        img_hat = ndimage.binary_dilation(img_hat, structure=np.ones((dilation_radius, dilation_radius)))
        plt.imshow(np.stack(((img_hat * 255).astype(np.uint8), img_gt, np.zeros_like(img_hat)), axis=0).T)
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