"""
Minimal 3D visualization to sanity-check clustering and boxes.
This is intentionally simple (matplotlib) so it runs everywhere.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _aabb_to_corners(box: np.ndarray) -> np.ndarray:
    """6-vector [xmin,ymin,zmin,xmax,ymax,zmax] -> (8,3) corners."""
    x0, y0, z0, x1, y1, z1 = box
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=float)

def _aabb_edges(corners: np.ndarray):
    """Return list of edge segments (pairs of corners) for Line3DCollection."""
    # bottom square (0-1-2-3), top square (4-5-6-7), verticals
    edges_idx = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return [(corners[i], corners[j]) for i,j in edges_idx]

def plot_points_and_boxes(points: np.ndarray, labels: np.ndarray, boxes: np.ndarray,
                          title: str = "", save_path: str | None = None, elev: int = 18, azim: int = -60):
    """
    Quick 3D sanity plot:
      - points colored by cluster label (noise is gray)
      - AABB wireframes overlaid
    """
    # Build colors per label (noise = -1 is gray)
    unique = np.unique(labels)
    lab_to_color = {}
    cmap = plt.cm.get_cmap("tab20", max(1, len(unique)))
    c_idx = 0
    for lab in unique:
        if lab == -1:
            lab_to_color[lab] = (0.6, 0.6, 0.6, 0.4)  # gray-ish translucent
        else:
            lab_to_color[lab] = cmap(c_idx)
            c_idx += 1

    colors = np.array([lab_to_color[lab] for lab in labels])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=2, c=colors, depthshade=False)

    # Draw AABB wireframes
    for b in boxes:
        corners = _aabb_to_corners(b)
        segs = _aabb_edges(corners)
        lc = Line3DCollection(segs, linewidths=1.5)
        ax.add_collection3d(lc)

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1,1,1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.close(fig)
