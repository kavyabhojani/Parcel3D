"""
Baseline clustering + AABB extraction.

What to say:
- DBSCAN groups dense regions without predefining number of clusters.
- We auto-suggest a global epsilon via the k-distance heuristic (robust percentile).
- Day 2 we’ll add Density-Aware ε per tile and OBB boxes.
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def suggest_eps_kdist(points: np.ndarray, k: int = 12, subsample: int = 6000, percentile: float = 80.0, seed: int = 0) -> float:
    """
    Suggest a global eps (ε) using the k-distance heuristic.

    Steps:
      1) For each sampled point, compute distance to its k-th nearest neighbor.
      2) Take a robust percentile (e.g., P80) as ε, which balances under/over-clustering.

    Tuning tips:
      - k often equals min_samples.
      - After voxelization, ε relates roughly to mean point spacing.
    """
    n = len(points)
    if n == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        X = points[idx]
    else:
        X = points

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="kd_tree").fit(points)
    dists, _ = nbrs.kneighbors(X, n_neighbors=k+1, return_distance=True)
    kd = dists[:, k]  # k-th neighbor distance (col 0 is self)
    eps = float(np.percentile(kd, percentile))
    if not np.isfinite(eps) or eps <= 0:
        eps = float(kd.mean())
    return eps


def dbscan_cluster(points: np.ndarray, eps: float, min_samples: int = 12) -> np.ndarray:
    """
    Run DBSCAN and return labels in [-1, 0..C-1], where -1 denotes noise.
    """
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1)
    labels = model.fit_predict(points)
    return labels


def extract_aabbs(points: np.ndarray, labels: np.ndarray, min_points: int = 20) -> np.ndarray:
    """
    Axis-aligned boxes per cluster (ignore noise and tiny clusters).
    Returns (M,6) where each row is [xmin, ymin, zmin, xmax, ymax, zmax].
    """
    assert points.shape[0] == labels.shape[0], "points and labels must align"
    boxes = []
    for lab in np.unique(labels):
        if lab < 0:   # -1 is DBSCAN noise
            continue
        idx = np.where(labels == lab)[0]
        if idx.size < min_points:
            continue
        p = points[idx]
        mins = p.min(axis=0)
        maxs = p.max(axis=0)
        boxes.append(np.r_[mins, maxs])

    if not boxes:
        return np.zeros((0, 6), dtype=float)
    boxes = np.stack(boxes, axis=0)
    # Sort by xmin just for readability in logs
    order = np.argsort(boxes[:, 0])
    return boxes[order]
