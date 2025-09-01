"""
Voxel-grid downsampling to stabilize density and speed up neighbor queries.

Why voxelize?
- DBSCAN is sensitive to point spacing; voxelizing evens that out.
- It also reduces N -> faster k-NN and clustering.
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


def voxel_downsample(points: np.ndarray, voxel_size: float, mode: str = "centroid") -> np.ndarray:
    """
    Snap points to a 3D grid of size `voxel_size` and keep one representative per voxel.

    mode:
      - 'centroid': average all points inside each voxel (better geometric fidelity)
      - 'first':    keep first seen point (slightly faster)

    Returns:
      points_ds: (M,3) downsampled points
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert voxel_size > 0
    if points.size == 0:
        return points

    # Integer voxel coordinate for each point
    vox = np.floor(points / voxel_size).astype(np.int64)

    # Robust unique by rows (Windows-friendly; avoids structured views)
    # uniq: (U,3) integer voxel coords, inv: (N,) mapping each point -> its voxel index
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)

    if mode == "first":
        # First point per voxel
        out = np.empty((len(uniq), 3), dtype=points.dtype)
        # Track first occurrence of each voxel index
        seen = np.full(len(uniq), False, dtype=bool)
        for i, g in enumerate(inv):
            if not seen[g]:
                out[g] = points[i]
                seen[g] = True
        return out

    # Centroid per voxel (default): scatter-add per axis (avoids broadcasting quirks)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    for d in range(3):
        np.add.at(sums[:, d], inv, points[:, d])

    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    # Guard against any zero counts (shouldn't happen, but keeps it safe)
    counts[counts == 0] = 1.0

    return (sums / counts[:, None]).astype(points.dtype)


def _mean_nn_spacing(points: np.ndarray, k: int = 1, max_samples: int = 5000, seed: int = 0) -> float:
    """
    Approximate mean distance to the k-th nearest neighbor.
    This is a quick health-check to report spacing after voxelization.
    """
    n = len(points)
    if n <= k:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    X = points[idx]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="kd_tree").fit(points)
    dists, _ = nbrs.kneighbors(X, n_neighbors=k+1, return_distance=True)
    return float(dists[:, k].mean())  # k-th neighbor (col 0 is the point itself)


def voxel_downsample_with_stats(points: np.ndarray, voxel_size: float, mode: str = "centroid", seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Downsample and return simple stats you can cite in the interview.
    """
    pts_ds = voxel_downsample(points, voxel_size=voxel_size, mode=mode)
    nn = _mean_nn_spacing(pts_ds, k=1, max_samples=5000, seed=seed)
    stats = {
        "input_count": int(points.shape[0]),
        "output_count": int(pts_ds.shape[0]),
        "reduction_ratio": float(pts_ds.shape[0] / max(1, points.shape[0])),
        "voxel_size": float(voxel_size),
        "approx_mean_nn_spacing": nn,
    }
    return pts_ds, stats
