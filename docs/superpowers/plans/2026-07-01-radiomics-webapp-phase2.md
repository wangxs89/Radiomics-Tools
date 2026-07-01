# 影像组学 Web App 第二阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完成可视化验证界面、完整特征提取流程、结果导出和高级模式功能

**Architecture:** 增强 Streamlit 应用，添加交互式 ROI 验证、图像预处理管道、完整特征提取和导出功能

**Tech Stack:** Streamlit, Plotly, pandas, SimpleITK, PyRadiomics

---

## Task 1: 可视化验证界面

**Files:**
- Modify: `app.py`
- Create: `src/roi_visualizer.py`

**目标：** 在上传 DICOM 和 RTSTRUCT 后，显示交互式影像查看器，让用户验证 ROI 位置

- [ ] **Step 1: 创建 ROI 可视化器**

创建 `src/roi_visualizer.py`，实现：
- 将 RTSTRUCT 轮廓转换为 2D 切片上的点
- 使用 Plotly 在影像上叠加 ROI 轮廓
- 支持多 ROI 显示（不同颜色）

- [ ] **Step 2: 更新 app.py 添加验证步骤**

在 `beginner_mode()` 中添加：
- 上传后显示影像查看器
- 显示 ROI 列表（可勾选）
- "确认无误" 按钮

- [ ] **Step 3: 测试和提交**

---

## Task 2: 完整特征提取管道

**Files:**
- Modify: `src/feature_extractor.py`
- Modify: `app.py`

**目标：** 实现完整的特征提取流程，从 DICOM + RTSTRUCT 到特征矩阵

- [ ] **Step 1: 创建 DICOM 到 SimpleITK 转换**

在 `feature_extractor.py` 中添加：
- `convert_dicom_to_sitk(dicom_series)` - 将 DICOM 序列转换为 SimpleITK Image
- `convert_rtstruct_to_mask(rtstruct, roi, dicom_series)` - 将 RTSTRUCT ROI 转换为掩模

- [ ] **Step 2: 实现批量提取**

添加 `extract_features_for_rois(image, masks_dict)` 方法，支持一次提取多个 ROI 的特征

- [ ] **Step 3: 集成到 app.py**

连接上传 → 验证 → 提取的完整流程

- [ ] **Step 4: 测试和提交**

---

## Task 3: 结果导出功能

**Files:**
- Create: `src/results_exporter.py`
- Modify: `app.py`

**目标：** 实现 CSV/Excel 导出，包含特征矩阵和元数据

- [ ] **Step 1: 创建结果导出器**

创建 `src/results_exporter.py`，实现：
- `export_to_csv(features_df, file_path)` - 导出 CSV
- `export_to_excel(features_df, metadata, file_path)` - 导出 Excel（带格式）
- 添加特征分类标签

- [ ] **Step 2: 集成下载按钮**

在 app.py 中添加：
- 特征表格预览
- CSV/Excel 下载按钮
- 显示特征统计信息

- [ ] **Step 3: 测试和提交**

---

## Task 4: 图像预处理模块

**Files:**
- Create: `src/image_preprocessor.py`
- Modify: `app.py`

**目标：** 实现图像预处理选项（重采样、归一化、灰度离散化）

- [ ] **Step 1: 创建预处理器**

创建 `src/image_preprocessor.py`，实现：
- `resample_image(image, new_spacing)` - 重采样
- `normalize_image(image, method)` - 归一化
- `discretize_image(image, bin_width)` - 灰度离散化

- [ ] **Step 2: 添加预处理配置 UI**

在高级模式中添加：
- 重采样参数输入
- 归一化选项
- 灰度离散化设置

- [ ] **Step 3: 集成到提取流程**

在特征提取前应用预处理

- [ ] **Step 4: 测试和提交**

---

## Task 5: 滤波器模块

**Files:**
- Create: `src/filters.py`
- Modify: `app.py`

**目标：** 实现 PyRadiomics 支持的滤波器（LoG、Wavelet 等）

- [ ] **Step 1: 创建滤波器模块**

创建 `src/filters.py`，实现：
- 滤波器配置生成器
- 支持 Original、LoG、Wavelet、Square、SquareRoot 等

- [ ] **Step 2: 添加滤波器选择 UI**

在高级模式中添加：
- 滤波器复选框
- LoG sigma 参数
- Wavelet 类型选择

- [ ] **Step 3: 集成到特征提取**

配置 PyRadiomics 使用选中的滤波器

- [ ] **Step 4: 测试和提交**

---

## Task 6: 高级模式完整实现

**Files:**
- Modify: `app.py`

**目标：** 完成高级模式的所有功能

- [ ] **Step 1: 实现参数面板**

添加：
- 图像预处理设置
- 滤波器选择
- 特征类型选择（形状、一阶、纹理等）
- 自定义参数

- [ ] **Step 2: 实现高级提取**

连接所有模块，支持：
- 自定义预处理
- 多滤波器组合
- 特征类型筛选

- [ ] **Step 3: 测试和提交**

---

## Task 7: 可视化报告

**Files:**
- Create: `src/report_generator.py`
- Modify: `app.py`

**目标：** 生成可视化报告（图表、热力图、统计信息）

- [ ] **Step 1: 创建报告生成器**

创建 `src/report_generator.py`，实现：
- 特征分布直方图
- 相关性热力图
- ROI 对比箱线图
- 特征统计摘要

- [ ] **Step 2: 集成到应用**

在结果页面添加：
- 可视化图表标签页
- 下载图表功能
- 统计信息展示

- [ ] **Step 3: 测试和提交**

---

## 完成标准

- [ ] 可视化验证界面工作正常
- [ ] 完整特征提取流程可用
- [ ] CSV/Excel 导出功能正常
- [ ] 图像预处理选项可用
- [ ] 滤波器选择功能正常
- [ ] 高级模式完整实现
- [ ] 可视化报告生成正常
- [ ] 所有测试通过
