"""
Hungarian matching + simple detection metrics for 3D AABB boxes.

Flow:
- Build IoU matrix (pred x gt)
- Hungarian assignment maximizes total IoU (1:1)
- Report Precision/Recall/F1 at chosen IoU thresholds
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    assignments: List[Tuple[int, int, float]]  # list of (pred_idx, gt_idx, iou)
    thresholds: List[float]
    table: List[Dict[str, float]]              # [{'iou_thr':0.25,'tp':..,'fp':..,'fn':..,'prec':..,'rec':..,'f1':..}, ...]
    mIoU_matched: float                        # mean IoU over matched pairs (no thresholding)


def hungarian_on_iou(iou: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Maximize total IoU with Hungarian assignment (which minimizes cost).
    Returns (pred_idx, gt_idx, iou) for each matched pair (including low IoUs).
    """
    if iou.size == 0:
        return []
    cost = 1.0 - iou  # similarity -> cost
    r_ind, c_ind = linear_sum_assignment(cost)
    out: List[Tuple[int, int, float]] = []
    for p, g in zip(r_ind, c_ind):
        out.append((int(p), int(g), float(iou[p, g])))
    return out


def _metrics_from_assignments(assignments: List[Tuple[int,int,float]], n_pred: int, n_gt: int, thr: float) -> Dict[str, float]:
    """
    Count TP/FP/FN at a given IoU threshold from the 1:1 assignments.
    """
    tps = sum(1 for _, _, v in assignments if v >= thr)
    fps = n_pred - tps
    fns = n_gt - tps
    prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    rec  = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"iou_thr": thr, "tp": float(tps), "fp": float(fps), "fn": float(fns), "prec": prec, "rec": rec, "f1": f1}


def evaluate_detections(pred_boxes: np.ndarray, gt_boxes: np.ndarray, thresholds: List[float]) -> MatchResult:
    """
    Full evaluation: IoU matrix -> Hungarian -> metrics at thresholds.
    """
    if pred_boxes.size == 0 and gt_boxes.size == 0:
        return MatchResult(assignments=[], thresholds=list(thresholds), table=[{"iou_thr": t, "tp": 0, "fp": 0, "fn": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0} for t in thresholds], mIoU_matched=0.0)

    from .iou3d import iou_matrix_aabb3d
    iou = iou_matrix_aabb3d(pred_boxes, gt_boxes)
    pairs = hungarian_on_iou(iou)

    # mean IoU over matched pairs (diagnostic; independent of thresholds)
    mIoU = float(np.mean([v for _, _, v in pairs])) if pairs else 0.0

    table = [_metrics_from_assignments(pairs, n_pred=len(pred_boxes), n_gt=len(gt_boxes), thr=float(t)) for t in thresholds]
    return MatchResult(assignments=pairs, thresholds=[float(t) for t in thresholds], table=table, mIoU_matched=mIoU)
