# Parcel3D — Clustering-based 3D Object Detection (Small Clouds)

**Goal**  
Given a 3D point cloud `[N, 3]` with multiple objects, return **approximate bounding boxes** for each object:  
`[x_min, y_min, z_min, x_max, y_max, z_max]`.

**Approach (geometry-first, learning-ready)**  
Focused on efficient **spatial grouping** with density-based clustering and tight boxes:
- **Voxel downsampling** to stabilize density and speed up neighbors/search.
- **DBSCAN baseline** (global ε) to form clusters without predefining K.
- **Axis-Aligned Bounding Boxes (AABB)** per cluster as a fast baseline.
- Later commits: **Density-Aware ε (DA-Eps)**, **OBB via PCA**, **IoU sweep**, **tile-and-merge streaming**, **calibration to .toml**, and a **confidence score** per box.

**Why this stands out**
1. **Density-Aware ε (DA-Eps)** per tile (not just a single global ε).
2. **Dual box modes** (AABB baseline, OBB optional) with an automatic chooser.
3. **Calibration mode** that writes suggested params to a `.toml`.
4. **Confidence score** per box (compactness × count × density ratio).
5. **IoU threshold sweep** (0.25 / 0.35 / 0.5) to show operating-point awareness.
6. **Tile-and-merge** sketch for streaming/real-time thinking.

---

### 1) Install
```bash
pip install numpy scikit-learn
