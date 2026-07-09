# Multi-DIC 架构草案

本项目根目录保存正式实现；`reference_code_lib/` 只作为本地参考源码，不进入正式项目。

## 当前入口

```powershell
python -m multidic run --config configs/MDIC.yaml --step validate
```

`configs/MDIC.yaml` 是主配置入口。`project.output_root` 按约定解释为 `project.case_root` 下的相对结果目录。

## 目录职责

- `configs/`: 用户可修改的流程配置。
- `multidic/`: Python 主流程，负责配置解析、数据校验、COLMAP 调用、DIC 调度、三维重建和后处理编排。
- `native/ncorr/`: 后续从参考 ncorr 源码迁移并去 MEX 化的 DIC 核心。
- `native/geometry/`: 后续放投影、三角化、尺度修正、相机融合等 C++ 几何核心。
- `native/bindings/`: 后续放 pybind11 绑定。
- `case/`: 示例或用户算例。
- `docs/`: 面向开发和使用的说明文档。
- `tests/`: 后续放配置、数据校验、几何和 DIC 的自动测试。

## 分步路线

1. 配置读取和算例校验。
2. COLMAP 工作区生成、运行和稀疏重建结果解析。
3. 棋盘格尺度修正到世界尺度。
4. 2D DIC 占位结果格式打通。
5. ncorr 核心按模块迁移、去 MEX 化并绑定到 Python。
6. MultiDIC 风格多相机三维重建、融合、位移和应变输出。
