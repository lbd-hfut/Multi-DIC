# Ncorr 二维匹配模块设计

本文档记录 Multi-DIC 中 `dic2d` 模块的 Ncorr 移植设计。目标是保留 Ncorr 原版 C++ 核心算法结构，去掉 MATLAB GUI 交互，并封装成可由配置文件驱动的二维匹配函数。

## 设计原则

- `reference_code_lib/ncorr_2D_matlab/` 只作为参考源码，不在运行时调用。
- `native/ncorr/` 承载正式移植代码。
- 原版 Ncorr 的 C++ 核心文件尽量少改，保持模块边界和算法流程可追溯。
- 原版 MATLAB GUI 负责的交互逻辑改写为无界面参数、ROI、seed 输入。
- `dic2d` 只做二维匹配和位移输出，不计算 Ncorr strain。

## 源码迁移边界

优先迁移并保留结构的 C++ 核心：

- `standard_datatypes.cpp/.h`
- `ncorr_datatypes.cpp/.h`
- `ncorr_lib.cpp/.h`
- `ncorr_alg_calcseeds.cpp`
- `ncorr_alg_rgdic.cpp`
- 后续如需要格式转换，可再引入 `ncorr_alg_convert.cpp`、`ncorr_alg_adddisp.cpp` 等非 GUI 算法文件。

需要替换的 MATLAB/GUI 层：

- `ncorr.m` 中的菜单、状态机、图窗刷新和 `setappdata/getappdata` 逻辑。
- `ncorr_gui_setdicparams.m` 改为 `DIC2DConfig`。
- `ncorr_gui_setrois.m` 改为 ROI mask 读取/生成规则。
- `ncorr_gui_setseeds.m` 改为 COLMAP 观测点自动 seed。
- `ncorr_gui_formatdisp.m` 中的单位、相关系数阈值、lenscoef 改为配置参数。
- `ncorr_gui_setstrainradius.m` 和 strain 相关算法不进入 `dic2d`。

## 函数接口

核心函数以单参考图、单变形图为最小调用单元：

```cpp
NcorrResult run_ncorr_dic2d(
    const GrayImage& reference_image,
    const GrayImage& deformed_image,
    const BoolMask& roi_mask,
    const SeedPoint& seed_point,
    const DIC2DConfig& config);
```

建议结果结构包含：

- 参考网格点 `reference_points`
- 匹配后的当前图点 `deformed_points`
- 位移 `u, v`
- affine 位移梯度 `ux, uy, vx, vy`
- 相关系数 `corrcoef`
- 有效点 mask
- Ncorr reduced-grid ROI
- `spacing + 1` 对应的像素尺度因子

Python 层 `pymultidic` API 负责组织相机、帧和文件输出；native 层只负责 Ncorr 算法计算。

## 配置字段

配置入口为 `dic2d`：

```yaml
dic2d:
  engine: ncorr
  analysis_type: regular
  subset_radius: 20
  subset_spacing: 5
  seed_search_radius: 50
  cutoff_diffnorm: 1.0e-6
  cutoff_iteration: 50
  num_threads: 1
  subset_truncation: false
  step_analysis:
    enabled: false
    type: seed
    auto: true
    step: 5
  roi:
    source: auto_or_user
    user_roi_mode: last_image
    mask_output_dir: masks/mask
    external_threshold: 127
    min_region_area: 2000
  seed:
    source: colmap_observations
    selection: first_inside_roi
  format:
    units_per_pixel: 1.0
    units: pixels
    cutoff_corrcoef: 0.6
    lenscoef: 0.0
  strain:
    enabled: false
```

`analysis_type: regular` 表示从参考图追踪到当前图。`dic2d` 暂不实现 backward 和 strain；字段保留是为了和 Ncorr 源概念对齐。

`seed_search_radius` 是当前 no-GUI 包装层用于 seed 初值 NCC 搜索的窗口半径，避免在大图上做全图搜索。后续接入原版 RG-DIC 传播后，该字段仍可作为自动 seed 初始化的保护参数。

`rg_search_radius` 是当前 reduced-grid 传播时每个相邻网格点围绕上一点位移初值做局部 NCC 搜索的半径。当前实现已经按 Ncorr RG-DIC 的四邻域扩张流程输出整张 reduced-grid 位移场，并在 NCC 初值后执行六参数 affine IC-GN 细化。后续若需要进一步贴近原版 Ncorr，可把当前 bilinear interpolation 替换为 biquintic interpolation。

## ROI 选择规则

每个相机的 ROI 按以下顺序确定：

1. 如果用户提供 ROI 图像，则使用相机文件夹中的最后一张图作为 ROI 图像。
2. 如果用户没有提供 ROI 图像，则使用 `mask` 步骤输出的自动 mask：
   `case/<case>/results/masks/mask/<cam_name>_mask.png` 或 `.npy`。
3. ROI 图像按 `dic2d.roi.external_threshold` 二值化。
4. 小连通区域按 `dic2d.roi.min_region_area` 过滤；该值对应 Ncorr 原版 `set_roi(..., cutoff=2000)` 的语义。

ROI mask 必须和参考图尺寸一致。

## Seed 选择规则

seed 来自 SfM/COLMAP 第一模块导出的当前相机观测点：

```text
case/<case>/results/sfm/<workspace>/observations.npz
```

使用字段：

- `cam_indices`
- `point_indices`
- `uv`

选择过程：

1. 按当前相机的 `cam_index` 过滤 `observations.uv`。
2. 只保留在图像边界内的点。
3. 只保留落在该相机 ROI mask 内的点。
4. 默认选择第一个满足条件的点作为 seed。
5. 如果没有点落在 ROI 内，`dic2d` 应报错并提示用户检查 ROI 或 SfM 观测点。

后续如需增强鲁棒性，可以把 `selection` 扩展为 `nearest_center_inside_roi` 或 `highest_track_length_inside_roi`。

## 输出约定

每个相机/帧对输出到：

```text
case/<case>/results/dic2d/
```

建议产物：

- `dic2d_<cam>_<frame>.npz`
  - `output_schema_version`: 输出契约版本，当前为 `2`
  - `x_ref`, `y_ref`: reduced-grid 参考图坐标
  - `x_def`, `y_def`: 匹配后的当前图坐标
  - `u`: reduced-grid x displacement
  - `v`: reduced-grid y displacement
  - `ux`, `uy`, `vx`, `vy`: affine 位移梯度
  - `corrcoef`: local NCC correlation
  - `valid`: valid reduced-grid points
  - `reduced_width`, `reduced_height`
  - `subset_radius`, `subset_spacing`
- `dic2d_report.json`

后续三维重建模块只依赖二维对应点、位移、相关系数和有效点 mask，不依赖 Ncorr GUI 数据结构。

## 不进入 dic2d 的内容

- Ncorr GUI 菜单、弹窗、手动 seed 图窗。
- Ncorr strain 半径设置和 Green-Lagrangian/Eulerian strain 图。
- 可视化交互控件。
- MATLAB `.mat` 作为内部主格式。

如后续需要应变，应在三维重建后单独实现 `strain` 模块。
