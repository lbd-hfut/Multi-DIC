# Multi-DIC 架构草案

`reference_code_lib/` 只作为本地参考源码，不进入正式项目实现。正式代码位于项目根目录，并通过配置文件驱动整个流程。

## 当前入口

```bash
python -m multidic run --config configs/MDIC.yaml --step validate
python -m multidic run --config configs/MDIC.yaml --step sfm
python -m multidic run --config configs/MDIC.yaml --step scale
python -m multidic run --config configs/MDIC.yaml --step mask
python -m multidic run --config configs/MDIC.yaml --step dic2d
python -m multidic run --config configs/MDIC.yaml --step recon3d
```

## 配置约定

- `project.case_root` 指向算例目录。
- `project.output_root` 是 `case_root` 下的相对子目录。
- `data.speckle_dir` 下按相机分文件夹保存散斑图像。
- `data.calibration_dir` 下按相机分文件夹保存棋盘格图像。
- `colmap.workspace` 控制 SfM 输出目录名。
- `scale_correction` 控制棋盘格尺度矫正参数。
- `mask` 控制用户 ROI mask 目录和自动 ROI 生成参数。
- `dic2d` 控制 Ncorr 风格二维匹配参数；详细设计见 `docs/NCORR_DIC2D_DESIGN.md`。

## 当前模块

### validate

检查配置、算例目录、相机图像目录、参考帧、变形帧和输出目录是否可用。

### sfm

使用每个相机文件夹下的第一张参考散斑图像构造 COLMAP 输入，调用官方 `pycolmap` 完成特征提取、匹配和增量重建。输出按 NDeF-DIC 风格保存到：

```text
case/<case_name>/results/sfm/colmap/
```

主要产物包括 `cameras.npz`、`cameras.mat`、`points3D.mat`、`sparse_points.npz`、`observations.npz` 以及相机和稀疏点分布图。

### scale

参照 NDeF-DIC 的 `sfm2world` 棋盘格流程，从 `cameras.npz` 读取 SfM 相机，检测 `calibrate_images/cam_*` 下的棋盘格角点，多相机对三角化棋盘角点，并根据物理棋盘格尺寸估计 `sfm_to_world_scale`。

主要产物包括：

```text
case/<case_name>/results/logs/scale_report.json
case/<case_name>/results/scale/sfm2world_scale.json
case/<case_name>/results/scale/chessboard_triangulation.npz
case/<case_name>/results/scale/detections/
```

### mask

参照 NDeF-DIC 的自动 ROI 流程生成每个相机视角下的 mask。若 `mask.user_mask_dir`
中已经提供完整的相机 ROI mask，则直接读取用户 mask；否则根据 SfM 观测点、Delaunay
支撑区域和参考散斑纹理自动生成 ROI。

主要产物包括：

```text
case/<case_name>/results/logs/mask_report.json
case/<case_name>/results/masks/mask/
case/<case_name>/results/masks/overlay/
case/<case_name>/results/masks/debug/
case/<case_name>/results/masks/auto_roi_meta.json
case/<case_name>/results/masks/auto_roi_summary.png
```

### dic2d

计划移植 `reference_code_lib/ncorr_2D_matlab` 中的 Ncorr 算法。移植策略是保留原版
C++ 核心结构，去掉 MATLAB GUI 外壳，将 Ncorr 封装为函数：

```text
run_ncorr_dic2d(reference_image, deformed_image, roi_mask, seed_point, dic2d_config)
```

ROI 优先使用用户提供的 ROI 图像；按当前约定，若用户提供 ROI，则每个相机文件夹下的
最后一张图作为 ROI 图像。若未提供 ROI，则使用 `mask` 步骤生成的自动 mask。seed
从 `sfm` 步骤导出的当前相机 COLMAP 观测点中选择，并要求 seed 位于 ROI mask 内。

`dic2d` 只负责二维匹配、相关系数和二维对应点输出，不计算 Ncorr strain。

## 后续模块

- `recon3d`：以 SfM track 为共同物理点锚点，将各相机 `dic2d` 位移插值到 COLMAP 观测点，再对参考帧和变形帧分别做多视角三角化，输出三维位移点云。
- `strain`：在三维重建后基于三维位移场计算全场应变，不放在 Ncorr `dic2d` 模块内。
- `visualization`：统一输出相机、点云、位移和应变可视化。

### recon3d

`recon3d` 第一版采用 sparse track anchored 设计。它不直接假设不同相机的 dense DIC 网格点互为同一物理点，而是使用 SfM/COLMAP 导出的 `point_indices` 作为跨视角对应关系：

```text
observations.npz 的同一 point_index
  -> 各相机参考图 uv
  -> 在 dic2d 位移场中插值得到 uv_def
  -> 多视角三角化 X_ref 与 X_def
  -> U = X_def - X_ref
```

输出位于：

```text
case/<case_name>/results/recon3d/
```

主要产物包括：

- `recon3d_<frame>.npz`
  - `point_indices`
  - `points_ref_sfm`, `points_def_sfm`, `displacement_sfm`
  - `points_ref_world`, `points_def_world`, `displacement_world`
  - `num_views`, `mean_corrcoef`
  - `reprojection_error_ref`, `reprojection_error_def`
  - `valid`
- `recon3d_<frame>.ply`
- `qc/<frame>/*_hist.png`
- `qc/<frame>/*_camera_contributions.png`
- `qc/<frame>/*_points_ref_colored.ply`
- `qc/<frame>/*_displacement_vectors.ply`
- `logs/recon3d_report.json`

计算后端优先尝试 `native_recon3d` pybind11 扩展；如果未编译，则使用 NumPy fallback。Python 层负责配置、IO、报告和导出；C++ 层负责批量 DIC 插值、多视角三角化和误差过滤。

QC 统计写入 `recon3d_report.json`，包括位移模长、参考/变形重投影误差、DIC 相关系数、有效视角数分布，以及每个相机贡献的 track 数量。QC 图和 PLY 默认随 `recon3d` 一起生成，可通过 `recon3d.qc` 和 `recon3d.export` 配置开关。

#### pair surface

为了对齐 MultiDIC 的 `DIC3DpairResults` 语义，`recon3d` 还会为每个 stereo pair 输出 pair surface：

```text
case/<case_name>/results/recon3d/pairs/<frame>/pair_<cam_a>_<cam_b>_<frame>.npz
```

默认 pair 选择规则为 `auto_spatial`：读取 `cameras.npz` 中的 `camera_centers_world`，用 PCA 将相机中心投影到主平面；若空间分布近似环绕、首尾空间距离不过大且共享 SfM track 足够，则按角度排序并首尾相连。若判定为非环绕布局，则只连接空间序列中的相邻相机，不把空间末端和起点强行组成 pair。若实验布局需要指定相机对，可在配置中改为手动：

```yaml
recon3d:
  pairs:
    mode: manual
    manual:
      - [cam_0, cam_2]
      - [cam_3, cam_5]
```

每个 pair surface 输出包含：

- `pair_cam_ids`
- `point_indices`
- `uv_ref_a`, `uv_ref_b`, `uv_def_a`, `uv_def_b`
- `points_ref_world`, `points_def_world`
- `displacement_world`, `displacement_norm_world`
- `corr_a`, `corr_b`, `corr_comb`
- `faces`, `valid_faces`
- `face_corr_comb`
- `face_centroids_ref`, `face_centroids_def`

其中 `corr_comb` 使用两个相机相关系数的较小值，表示该点的弱侧质量；`face_corr_comb` 使用三角形顶点质量的较小值，语义上对应 MultiDIC 中按 face 汇总的最差相关质量。

#### post3d

为了对齐 MultiDIC `STEP4_PostProcessing.m` 的位移场后处理，`recon3d` 会为每个 pair surface 生成：

```text
case/<case_name>/results/recon3d/post/<frame>/pair_<cam_a>_<cam_b>_<frame>_post.npz
```

当前 post3d 覆盖：

- `displacement_world`, `displacement_norm_world`
- `rbm_rotation`, `rbm_translation`
- `points_def_arbm_world`
- `displacement_arbm_world`, `displacement_arbm_norm_world`
- `face_centroids_ref`, `face_centroids_def`, `face_centroids_arbm`
- `face_corr_comb`
- `face_isotropy_index`
- `face_displacement_world`, `face_displacement_arbm_world`

刚体运动去除使用与 MultiDIC `rigidTransformation.m` 对齐的 SVD/Kabsch 刚体配准：将变形点云刚体对齐到参考点云，再计算 `ARBM` 位移。`face_isotropy_index` 使用 MultiDIC `faceIsotropyIndex.m` 的三角形协方差特征值公式。
