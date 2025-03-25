import numpy as np

# 3D metrics

def dice3d(gt, hat):
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
    return cd / 2

def iou3d(gt, hat):
    gt_set = set(map(tuple, gt))
    hat_set = set(map(tuple, hat))
    I = len(gt_set.intersection(hat_set))
    U = len(gt_set.union(hat_set))
    return I / U

# 2D metrics

def dice2d(img_gt, img_hat):
    A = img_gt != 0
    B = img_hat != 0
    intersection = np.sum(A & B)
    union = np.sum(A) + np.sum(B)
    return 2 * intersection / union
