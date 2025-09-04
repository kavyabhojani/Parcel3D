"""
Synthetic scene generator for small 3D point clouds.

Why:
- Need quick, controllable data to iterate on clustering logic.
- Scenes contain a few primitive objects (sphere, cube, cylinder) with noise/outliers.
- Output: points (N,3), ground-truth boxes (M,6) as AABB in world frame, and metadata.

- Surfaces are sampled directly (no meshes).
- Each object gets a random pose (rotation + translation).
- AABB (axis-aligned) of transformed points is the GT box.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np


@dataclass
class ObjSpec:
    kind: str                          # 'sphere' | 'cube' | 'cylinder'
    size: Tuple[float, ...]            # sphere: (radius,), cube: (edge,), cylinder: (radius, height)
    center: Tuple[float, float, float] # object center in world coords
    n_points: int                      # points to sample on the surface
    noise_std: float = 0.0             # Gaussian noise std added to sampled points
    rot_deg: Tuple[float, float, float] = (0, 0, 0)  # Euler XYZ rotation in degrees


#small helpers: rotations and rigid transform 
def _deg2rad(e):
    return np.deg2rad(e)

def _rot_xyz_matrix(deg_xyz: Tuple[float, float, float]) -> np.ndarray:
    """XYZ Euler rotation (degrees) → 3x3 rotation matrix."""
    rx, ry, rz = _deg2rad(deg_xyz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _apply_pose(points: np.ndarray, center, rot_deg) -> np.ndarray:
    """Apply rotation then translation to local points."""
    R = _rot_xyz_matrix(rot_deg)
    return (R @ points.T).T + np.asarray(center)[None, :]


#primitive surface samplers

def _sample_sphere_surface(radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random directions, normalize to unit sphere, scale by radius."""
    v = rng.normal(size=(n, 3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return radius * v

def _sample_cube_surface(edge: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample cube faces by picking a face then uniform (u,v) on that face."""
    half = edge / 2.0
    points = np.empty((n, 3), dtype=float)
    faces = rng.integers(0, 6, size=n)
    uv = rng.uniform(-half, half, size=(n, 2))
    for i, f in enumerate(faces):
        u, v = uv[i]
        if f == 0:   points[i] = [ half,  u,    v]
        elif f == 1: points[i] = [-half,  u,    v]
        elif f == 2: points[i] = [ u,   half,   v]
        elif f == 3: points[i] = [ u,  -half,   v]
        elif f == 4: points[i] = [ u,    v,   half]
        else:        points[i] = [ u,    v,  -half]
    return points

def _sample_cylinder_surface(radius: float, height: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample cylinder side (70%) + caps (30%) for variety."""
    n_side = int(0.7 * n)
    n_caps = n - n_side
    # side
    theta = rng.uniform(0, 2*np.pi, size=n_side)
    z = rng.uniform(-height/2.0, height/2.0, size=n_side)
    side = np.c_[radius*np.cos(theta), radius*np.sin(theta), z]
    # caps: uniform over disk area via sqrt on radius
    r = radius * np.sqrt(rng.uniform(0, 1, size=n_caps))
    ang = rng.uniform(0, 2*np.pi, size=n_caps)
    x, y = r*np.cos(ang), r*np.sin(ang)
    sign = rng.integers(0, 2, size=n_caps) * 2 - 1   # -1 or +1
    zc = (height/2.0) * sign
    caps = np.c_[x, y, zc]
    return np.vstack([side, caps])


#utilities

def _aabb(points: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax] for a set of points."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return np.r_[mins, maxs]

def make_scene_from_specs(specs: List[ObjSpec], outlier_ratio: float, outlier_scale: float, seed: int):
    """
    Compose a scene from object specs, add global outliers around the scene box.

    Returns:
        points (N,3), gt_boxes (M,6), meta (dict)
    """
    rng = np.random.default_rng(seed)
    pts_all, gt_boxes, meta = [], [], {"objects": []}

    for j, s in enumerate(specs):
        if s.kind == "sphere":
            (radius,) = s.size
            local = _sample_sphere_surface(radius, s.n_points, rng)
        elif s.kind == "cube":
            (edge,) = s.size
            local = _sample_cube_surface(edge, s.n_points, rng)
        elif s.kind == "cylinder":
            radius, height = s.size
            local = _sample_cylinder_surface(radius, height, s.n_points, rng)
        else:
            raise ValueError(f"Unknown kind: {s.kind}")

        if s.noise_std > 0:
            local = local + rng.normal(scale=s.noise_std, size=local.shape)

        world = _apply_pose(local, s.center, s.rot_deg)
        pts_all.append(world)
        gt_boxes.append(_aabb(world))
        meta["objects"].append({
            "idx": j, "kind": s.kind, "size": s.size, "center": s.center,
            "rot_deg": s.rot_deg, "n_points": s.n_points
        })

    points = np.vstack(pts_all)
    gt_boxes = np.vstack(gt_boxes)

    # Add uniform outliers around the whole scene to simulate dust/noise
    if outlier_ratio > 0:
        scene_box = _aabb(points)
        mins, maxs = scene_box[:3], scene_box[3:]
        span = (maxs - mins) * outlier_scale
        mins_o = mins - 0.1 * span
        maxs_o = maxs + 0.1 * span
        n_out = int(points.shape[0] * outlier_ratio)
        outliers = rng.uniform(mins_o, maxs_o, size=(n_out, 3))
        points = np.vstack([points, outliers])

    return points.astype(np.float32), gt_boxes.astype(np.float32), meta

#ready-made scenes for the first day

def make_synthetic_scene_easy(seed: int = 0):
    """Two separated objects, light noise, few outliers."""
    specs = [
        ObjSpec("sphere", (0.4,),  center=(-1.0, 0.2, 0.2), n_points=2000, noise_std=0.01, rot_deg=(20, 0, 15)),
        ObjSpec("cube",   (0.9,),  center=( 1.1,-0.1, 0.0), n_points=2500, noise_std=0.01, rot_deg=( 0,30,  0)),
    ]
    return make_scene_from_specs(specs, outlier_ratio=0.04, outlier_scale=1.1, seed=seed)

def make_synthetic_scene_dense(seed: int = 1):
    """Three closer objects, denser sampling, more outliers."""
    specs = [
        ObjSpec("sphere",   (0.35,),      center=(-0.7,  0.0,  0.0), n_points=4000, noise_std=0.012, rot_deg=(10,15,20)),
        ObjSpec("cube",     (0.8,),       center=( 0.6, -0.1,  0.0), n_points=4500, noise_std=0.012, rot_deg=( 0,35, 5)),
        ObjSpec("cylinder", (0.35, 1.0),  center=( 0.1,  0.7,  0.0), n_points=3800, noise_std=0.012, rot_deg=( 0, 0,30)),
    ]
    return make_scene_from_specs(specs, outlier_ratio=0.08, outlier_scale=1.2, seed=seed)

def save_scene_npz(path: str, points: np.ndarray, gt_boxes: np.ndarray, meta: Dict) -> None:
    """Persist a scene to disk as .npz (keeps repo tidy)."""
    np.savez(path,
             points=points.astype(np.float32),
             gt_boxes=gt_boxes.astype(np.float32),
             meta=np.array(meta, dtype=object))
