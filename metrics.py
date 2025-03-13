import numpy as np

# 3D metrics

def dice3d(gt, hat):
    print(gt.shape, hat.shape)
    distances = np.linalg.norm(gt[:, np.newaxis] - hat, axis=2)
    min_dist_gt_to_hat = np.min(distances, axis=1)
    intersection = np.sum(min_dist_gt_to_hat == 0)
    union = gt.shape[0] + hat.shape[0]
    return 2 * intersection / union

def chamfer3d(gt, hat):
    distances = np.linalg.norm(gt[:, np.newaxis] - hat, axis=2)
    min_dist_gt_to_hat = np.min(distances, axis=1)    
    min_dist_hat_to_gt = np.min(distances, axis=0)
    cd = np.mean(min_dist_gt_to_hat) + np.mean(min_dist_hat_to_gt)
    return cd

def cl_dice3d(gt, hat):
    return 0


def iou3d(gt, hat):
    pass

# 2D metrics

def dice2d(gt, hat):
    pass
