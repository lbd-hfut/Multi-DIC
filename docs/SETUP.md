# Multi-DIC 环境配置

当前项目的 COLMAP/SfM 阶段使用官方 `pycolmap` 接口，并固定安装
`pycolmap==4.1.0`，不再维护自写的 `native/colmap_backend`。

## WSL 环境

推荐在 WSL/Linux 下使用项目专用 micromamba 环境：

```bash
~/.local/bin/micromamba create -y -f environment.yml
~/.local/bin/micromamba run -n multi-dic python -m pip install -e .
```

也可以重新创建同名环境：

```bash
~/.local/bin/micromamba env remove -n multi-dic
~/.local/bin/micromamba create -y -f environment.yml
~/.local/bin/micromamba run -n multi-dic python -m pip install -e .
```

## 流程检查

```bash
~/.local/bin/micromamba run -n multi-dic python -m multidic run \
  --config configs/MDIC.yaml --step validate

~/.local/bin/micromamba run -n multi-dic python -m multidic run \
  --config configs/MDIC.yaml --step sfm

~/.local/bin/micromamba run -n multi-dic python -m multidic run \
  --config configs/MDIC.yaml --step scale

~/.local/bin/micromamba run -n multi-dic python -m multidic run \
  --config configs/MDIC.yaml --step mask
```

`sfm` 会按 NDeF-DIC 的流程读取 `case_root/images/cam_*` 下自然排序后的第一张图像，
复制到扁平的 `colmap_images/` 目录，然后运行 pycolmap 特征提取、穷举匹配和增量重建。
主相机由 `colmap.reference_camera` 指定，默认 `cam_0`。导出的 SfM 坐标系为：
原点在保留稀疏点云质心，坐标轴平行于主相机。

主要输出包括：

- `case/CylinderDIC/results/logs/sfm_report.json`
- `case/CylinderDIC/results/sfm/colmap/colmap.db`
- `case/CylinderDIC/results/sfm/colmap/colmap_images/`
- `case/CylinderDIC/results/sfm/colmap/colmap_sfm/`
- `case/CylinderDIC/results/sfm/colmap/cameras.npz`
- `case/CylinderDIC/results/sfm/colmap/cameras.mat`
- `case/CylinderDIC/results/sfm/colmap/cameras.json`
- `case/CylinderDIC/results/sfm/colmap/sparse_points.npz`
- `case/CylinderDIC/results/sfm/colmap/points3D.mat`
- `case/CylinderDIC/results/sfm/colmap/observations.npz`
- `case/CylinderDIC/results/sfm/colmap/sparse_scene.png`
- `case/CylinderDIC/results/sfm/colmap/camera_observations_3d.png`
- `case/CylinderDIC/results/sfm/colmap/camera_observations_2d.png`

`scale` 会按 NDeF-DIC 的 `sfm2world` 棋盘格流程读取 `cameras.npz`，在
`calibrate_images/cam_*` 中检测棋盘格角点，三角化棋盘格角点，并估计
`sfm_to_world_scale`。主要输出包括：

- `case/CylinderDIC/results/logs/scale_report.json`
- `case/CylinderDIC/results/scale/sfm2world_scale.json`
- `case/CylinderDIC/results/scale/chessboard_triangulation.npz`
- `case/CylinderDIC/results/scale/detections/`

`mask` 会先检查 `mask.user_mask_dir` 指定的目录。若该目录下已经有每个相机对应的
ROI mask，则直接读取用户 mask；若没有提供完整 mask，则按 NDeF-DIC 的自动 ROI
流程，根据 SfM 观测点和参考散斑图像纹理生成每个相机视角下的 ROI。

用户 mask 支持 `.npy` 和常见图像格式，推荐命名为：

- `cam_0_mask.png`
- `cam_1_mask.png`
- `cam_2_mask.npy`

自动或外部 mask 都会统一输出到：

- `case/CylinderDIC/results/logs/mask_report.json`
- `case/CylinderDIC/results/masks/mask/`
- `case/CylinderDIC/results/masks/overlay/`
- `case/CylinderDIC/results/masks/debug/`
- `case/CylinderDIC/results/masks/auto_roi_meta.json`
- `case/CylinderDIC/results/masks/auto_roi_summary.png`
