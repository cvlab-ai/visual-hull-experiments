#!.test-env/bin/python3

import matplotlib.pyplot as plt

from xray_angio_3d import reconstruction
from util import *

def figure_2_synth():
    gt, xinfs = load_sample(0, "tortuous")
    reconstructed = reconstruction(xinfs)
    reconstructed['vessel'] = voxelize_points(np.array(reconstructed['vessel']))
    gt = voxelize_points(gt)
    print(reconstructed.keys())
    X_gt, Y_gt, Z_gt = zip(*gt)
    X_hat, Y_hat, Z_hat = zip(*reconstructed['vessel'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_gt, Y_gt, Z_gt, c='r', marker='o')
    ax.scatter(X_hat, Y_hat, Z_hat)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('GT')
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("tkagg")
    # matplotlib.rcParams['webagg.address'] = '0.0.0.0'
    figure_2_synth()
