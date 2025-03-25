import os
import pickle
import numpy as np
import argparse
import yaml

INPUT_DIR="data/"
VOXELIZATION_THRESH_DEFAULT=0.003

def voxelize_points(points, voxel_size=VOXELIZATION_THRESH_DEFAULT):
    scaled_points = np.floor(points / voxel_size).astype(int)
    unique_voxels = np.unique(scaled_points, axis=0)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2.0
    return voxel_centers

def load_sample(experiment_name, ix, tortosity):
    try:
        tortosity_dir = os.path.join(INPUT_DIR, experiment_name, str(tortosity))
        gt_filename = os.path.join(tortosity_dir, f'{ix}.gt')
        xinfs_filename = os.path.join(tortosity_dir, f'{ix}.xinfs')
        noise_filename = os.path.join(tortosity_dir, f'{ix}.noise')
        with open(gt_filename, 'rb') as gt_file:
            gt = pickle.load(gt_file)
        with open(xinfs_filename, 'rb') as xinfs_file:
            xinfs = pickle.load(xinfs_file)
        with open(noise_filename, 'rb') as f:
            noise = pickle.load(f)
    except:
        gt, xinfs, noise = None, None, None
    return gt, xinfs, noise

def load_config():
    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    with open(args.file, "r") as file:
        config = yaml.safe_load(file)
    return config


def convert_clinical_to_standard_angles(clinical_angles):
    '''
    clinical_angles: A list of strings; each string must have the form "LAO/RAO X, CRA/CAU Y".
    For example, ['LAO 35, CAU 10', 'RAO 20, CRA 0']

    This function assumes that the clinical angles refer to a patient lying supine in the xz plane.
    Standard angles refer to azimuthal and elevation angle representation of spherical coordinates.
    '''
    clinical_key = {'RAO': 1, 'LAO': -1, 'CRA': 1, 'CAU': -1}
    theta_array = []
    phi_array = []
    for angle_pair in clinical_angles:
        theta_string = angle_pair.split(',')[0]
        phi_string = angle_pair.split(',')[1]

        theta_clinical = float(theta_string.split()[1])*clinical_key[theta_string.split()[0]]
        phi_clinical = float(phi_string.split()[1])*clinical_key[phi_string.split()[0]]

        theta = theta_clinical - 90
        phi = phi_clinical

        theta_array.append(theta)
        phi_array.append(phi)

    return theta_array, phi_array
    
print(
convert_clinical_to_standard_angles([
    "RAO 30, CRA 30",
    "LAO 60, CRA 30",
    "RAO 30, CAU 30"
])
)

# print(
# convert_clinical_to_standard_angles([
#     "LAO 40, CAU 30",
#     "LAO 60, CRA 30",
#     "RAO 30, CAU 30"
# ])
# )
