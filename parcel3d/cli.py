"""
Small CLI to:
  1) generate synthetic scenes
  2) run baseline detection (voxel -> suggest eps -> DBSCAN -> AABB)
  3) optionally plot and/or save a figure, and print evaluation metrics

Run examples:
  python -m parcel3d.cli generate --scene easy
  python -m parcel3d.cli detect examples/scene_easy.npz --voxel 0.05 --min-samples 12 --metrics --plot --save figs/easy_eval.png
"""

from __future__ import annotations
import argparse, os
import numpy as np

from .data.synthetic import (
    make_synthetic_scene_easy,
    make_synthetic_scene_dense,
    save_scene_npz,
)
from .pipeline.voxel import voxel_downsample_with_stats
from .pipeline.cluster import suggest_eps_kdist, dbscan_cluster, extract_aabbs
from .viz.plot3d import plot_points_and_boxes


def _cmd_generate(args: argparse.Namespace) -> int:
    os.makedirs("examples", exist_ok=True)

    if args.scene == "easy":
        pts, gt_boxes, meta = make_synthetic_scene_easy(seed=args.seed)
        out = os.path.join("examples", "scene_easy.npz")
    else:
        pts, gt_boxes, meta = make_synthetic_scene_dense(seed=args.seed)
        out = os.path.join("examples", "scene_dense.npz")

    save_scene_npz(out, pts, gt_boxes, meta)
    print(f"[generate] wrote {out}  (points={len(pts):,}, gt_boxes={len(gt_boxes)})")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    # ---- load scene ----
    data = np.load(args.input, allow_pickle=True)
    points = data["points"]
    gt_boxes = data["gt_boxes"] if "gt_boxes" in data.files else np.zeros((0, 6), dtype=np.float32)
    print(f"[detect] loaded {args.input}  points={len(points):,}  gt_boxes={len(gt_boxes)}")

    # ---- voxel downsample ----
    pts_ds, stats = voxel_downsample_with_stats(points, voxel_size=args.voxel, seed=args.seed)
    print(f"[voxel] input={stats['input_count']:,}  output={stats['output_count']:,}  "
          f"ratio={stats['reduction_ratio']:.3f}  mean_nn≈{stats['approx_mean_nn_spacing']:.4f} m")

    # ---- eps selection ----
    if args.eps is not None:
        eps = float(args.eps)
        print(f"[eps] using user-provided eps={eps:.4f}")
    else:
        eps = suggest_eps_kdist(pts_ds, k=args.min_samples, subsample=args.subsample,
                                percentile=args.percentile, seed=args.seed)
        print(f"[eps] suggest_eps_kdist: k={args.min_samples} percentile={args.percentile} → eps≈{eps:.4f}")

    # ---- clustering ----
    labels = dbscan_cluster(pts_ds, eps=eps, min_samples=args.min_samples)
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(np.sum(np.unique(labels) >= 0))
    print(f"[dbscan] clusters={n_clusters}  noise_points={n_noise:,}")

    # ---- boxes (AABB) ----
    boxes = extract_aabbs(pts_ds, labels, min_points=args.min_points)
    print("[boxes] AABB (xmin, ymin, zmin, xmax, ymax, zmax):")
    for i, b in enumerate(boxes):
        print(f"  #{i}: " + " ".join(f"{v:.3f}" for v in b))

    # ---- evaluation (AABB) ----
    if args.metrics:
        from .eval.match import evaluate_detections  # local import to keep CLI fast when metrics aren't requested
        res = evaluate_detections(boxes, gt_boxes, thresholds=list(args.iou_thresholds))
        print(f"[eval] mIoU over matched pairs: {res.mIoU_matched:.3f}")
        print("[eval] IoU threshold sweep:")
        for row in res.table:
            print(f"  IoU≥{row['iou_thr']:.2f}  TP={int(row['tp'])}  FP={int(row['fp'])}  FN={int(row['fn'])}  "
                  f"Prec={row['prec']:.3f}  Rec={row['rec']:.3f}  F1={row['f1']:.3f}")

    # ---- optional plot ----
    if args.plot or args.save:
        title = f"AABB baseline — eps={eps:.3f}, min_samples={args.min_samples}, voxel={args.voxel}"
        save_path = args.save if args.save else None
        plot_points_and_boxes(pts_ds, labels, boxes, title=title, save_path=save_path, gt_boxes=gt_boxes)
        if save_path:
            print(f"[plot] saved figure to {save_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parcel3d", description="Parcel3D: clustering-based 3D object detection (baseline).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # generate
    g = sub.add_parser("generate", help="generate a synthetic scene (.npz)")
    g.add_argument("--scene", choices=["easy", "dense"], default="easy")
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=_cmd_generate)

    # detect
    d = sub.add_parser("detect", help="run baseline detection on a .npz scene")
    d.add_argument("input", type=str, help="path to .npz scene (e.g., examples/scene_easy.npz)")
    d.add_argument("--voxel", type=float, default=0.05, help="voxel size in same units as scene (meters assumed)")
    d.add_argument("--min-samples", type=int, default=12, help="DBSCAN min_samples; often matches k in k-dist")
    d.add_argument("--percentile", type=float, default=80.0, help="percentile for k-distance eps suggestion")
    d.add_argument("--subsample", type=int, default=6000, help="subsample size for eps suggestion")
    d.add_argument("--eps", type=float, default=None, help="override eps (skip suggestion)")
    d.add_argument("--min-points", type=int, default=30, help="drop clusters with fewer points than this")
    d.add_argument("--plot", action="store_true", help="show a 3D plot")
    d.add_argument("--save", type=str, default=None, help="save plot to this path (e.g., figs/easy_baseline.png)")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--metrics", action="store_true", help="compute PR/F1 and mIoU (AABB)")
    d.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.5],
                   help="IoU thresholds for the sweep (e.g., 0.25 0.5)")
    d.set_defaults(func=_cmd_detect)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
