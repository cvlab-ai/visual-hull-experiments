import numpy as np

# 3D metrics

def dice3d(gt, hat):
    print(gt.shape, hat.shape)
    distances = np.linalg.norm(gt[:, np.newaxis] - hat, axis=2)
    min_dist_gt_to_hat = np.min(distances, axis=1)
    intersection = np.sum(min_dist_gt_to_hat == 0)
    union = gt.shape[0] + hat.shape[0]
    return 2 * intersection / union

def cl_dice3d(gt, hat):
    pass

def chamfer_dist(gt, hat):
    pass

def iou3d(gt, hat):
    pass

# 2D metrics

def dice2d(gt, hat):
    pass
