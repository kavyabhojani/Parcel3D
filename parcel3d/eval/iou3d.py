"""
3D IoU for axis-aligned boxes (AABB).
Boxes are [xmin, ymin, zmin, xmax, ymax, zmax].

This is vectorized: (M,6) vs (K,6) -> (M,K) IoU matrix.
"""

from __future__ import annotations
import numpy as np


def iou_matrix_aabb3d(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of AABBs.

    Args:
        pred_boxes: (M,6)
        gt_boxes:   (K,6)
    Returns:
        iou: (M,K) with values in [0,1]
    """
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=float)

    # Expand dims to broadcast: P -> (M,1,6), G -> (1,K,6)
    P = pred_boxes[:, None, :]
    G = gt_boxes[None, :, :]

    # Intersection box edges (axis-wise overlap)
    mins = np.maximum(P[..., :3], G[..., :3])   # (M,K,3)
    maxs = np.minimum(P[..., 3:], G[..., 3:])   # (M,K,3)
    inter_dims = np.clip(maxs - mins, a_min=0.0, a_max=None)
    inter_vol = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]

    # Volumes for P and G
    vol_p = np.prod(P[..., 3:] - P[..., :3], axis=-1)   # (M,1)
    vol_g = np.prod(G[..., 3:] - G[..., :3], axis=-1)   # (1,K)

    union = vol_p + vol_g - inter_vol
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter_vol / union, 0.0)

    return np.clip(iou, 0.0, 1.0)
