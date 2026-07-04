# Radiomics Tools

Radiomics Tools 是一个基于 Streamlit、PyRadiomics、SimpleITK 和 pydicom 的医学影像组学/剂量组学 Web 应用。它面向放疗、影像组学和临床预测建模场景，支持从 DICOM 影像、RTSTRUCT 勾画和 RTDOSE 剂量文件中提取特征，并将病例级结果保存到临时数据库后用于统计分析和机器学习建模。

> 重要说明：本工具用于科研数据整理、特征提取和建模探索，不应直接作为临床诊疗决策依据。

## 主要功能

- Beginner 模式：选择病例文件夹、验证 ROI、提取影像组学和剂量组学特征。
- Advanced 模式：控制预处理、滤波器、特征类别、灰度离散化和 2D/3D 提取参数。
- 多影像序列分析：一个病例文件夹中存在多个 CT/MR/PET 等影像序列时，可分别提取并在结果中保留 Series 信息。
- RTSTRUCT ROI 可视化：在提取前检查 ROI 位置、切片和窗宽窗位。
- 剂量组学：读取 RTDOSE，将 ROI mask 重采样到剂量网格并提取剂量影像特征。
- 临时数据库：将影像组学/剂量组学结果保存到本地 SQLite 数据库 `.radiomics_tmp/radiomics_results.sqlite3`。
- 临床随访导入：上传 CSV/XLSX 随访表，通过 Case ID 与组学特征合并。
- 建模数据集构建：自动生成一例一行的宽表，用于回归、分类和聚类。
- 机器学习建模：自动划分 training、validation、test，输出验证集和测试集评价指标。
- 统计分析：相关性分析、LASSO 特征筛选、ICC 可靠性分析。
- 结果导出：支持 CSV、Excel 下载。

## 支持的数据

推荐将同一病例的相关 DICOM 文件放在同一个文件夹中：

```text
Case_001/
  CT*.dcm 或 MR*.dcm
  RTSTRUCT*.dcm
  RTDOSE*.dcm          # 可选
```

应用会自动扫描：

- CT、MR、PT/PET 等影像序列；
- RTSTRUCT 结构文件；
- RTDOSE 剂量文件；
- 其他 DICOM 序列会显示为 skipped；
- NIfTI mask 当前仅做发现提示，完整支持尚未启用。

## 安装

建议使用 Python 3.9 或更新版本。

```bash
git clone https://github.com/wangxs89/Radiomics-Tools.git
cd Radiomics-Tools
python3 -m pip install -r requirements.txt
```

macOS ARM64 如遇到 PyRadiomics 构建问题，可尝试：

```bash
python3 -m pip install versioneer
python3 -m pip install pyradiomics --no-build-isolation
```

## 启动

```bash
streamlit run app.py
```

默认浏览器地址通常为：

```text
http://localhost:8501
```

如果端口被占用，Streamlit 会自动使用其他端口，例如 `8502`。

## 快速使用流程

1. 打开应用后选择 Beginner 或 Advanced 模式。
2. 在 Folder path 中输入病例文件夹路径，或点击 Browse Folder。
3. 选择影像序列。如果一个病例包含多个影像序列，后续可选择多个序列分别提取。
4. 在 ROI 可视化区域确认勾画位置。
5. 选择需要提取的 ROI。
6. 输入或确认 Case ID。
7. 点击 Extract All Features。
8. 在页面下方查看特征矩阵，并下载 CSV/Excel。
9. 如需后续建模，点击 Save Imaging Features 或 Save Dose Features 保存到临时数据库。
10. 进入 Database & Modeling 页面，上传临床随访表并创建预测模型。

## Beginner 模式

Beginner 模式适合快速提取：

- 自动读取 DICOM 影像、RTSTRUCT 和 RTDOSE；
- 自动使用默认 PyRadiomics 特征类别；
- 支持多影像序列分别提取；
- 结果中包含 CaseID、Series、SeriesUID、Modality、FeatureKind、ROI 等元数据；
- 如果存在 RTDOSE，会对所选 ROI 同时提取剂量特征。

## Advanced 模式

Advanced 模式提供更多控制项：

- Image Preprocessing：重采样、归一化、离散化；
- Filter Selection：选择 Original、LoG、Wavelet、Gradient、LBP 等图像类型；
- Feature Types：选择 shape、shape2D、firstorder、GLCM、GLRLM、GLSZM、GLDM、NGTDM；
- PyRadiomics Settings：
  - `binWidth`：灰度离散化 bin width；
  - `resampledPixelSpacing`：PyRadiomics 内部重采样 spacing；
  - `force2D`：强制按 2D 切片提取；
  - `force2Ddimension`：指定 2D 方向；
- 勾选 Shape 2D 时，应用会自动启用 force2D。

## 影像组学特征类别

本应用使用 PyRadiomics 的标准特征体系。特征名通常形如：

```text
original_glcm_Contrast
wavelet-LLH_firstorder_Mean
gradient_glrlm_RunEntropy
```

命名结构一般为：

```text
图像类型_特征类别_特征名
```

### 非纹理类特征

| 类别 | 含义 | 典型用途 |
| --- | --- | --- |
| `firstorder` | ROI 内灰度/强度直方图统计，不考虑空间关系 | 描述 CT HU、MR/PET 强度或剂量分布 |
| `shape` | 3D 形状特征 | 描述体积、表面积、球形度、轴长等 |
| `shape2D` | 2D 形状特征 | 单层或强制 2D 分析 |

`firstorder` 当前可提取 19 个参数：

```text
10Percentile, 90Percentile, Energy, Entropy, InterquartileRange,
Kurtosis, Maximum, MeanAbsoluteDeviation, Mean, Median, Minimum,
Range, RobustMeanAbsoluteDeviation, RootMeanSquared, Skewness,
StandardDeviation, TotalEnergy, Uniformity, Variance
```

`shape` 当前可提取 17 个参数：

```text
Compactness1, Compactness2, Elongation, Flatness, LeastAxisLength,
MajorAxisLength, Maximum2DDiameterColumn, Maximum2DDiameterRow,
Maximum2DDiameterSlice, Maximum3DDiameter, MeshVolume, MinorAxisLength,
SphericalDisproportion, Sphericity, SurfaceArea, SurfaceVolumeRatio,
VoxelVolume
```

`shape2D` 当前可提取 10 个参数：

```text
Elongation, MajorAxisLength, MaximumDiameter, MeshSurface, MinorAxisLength,
Perimeter, PerimeterSurfaceRatio, PixelSurface, SphericalDisproportion,
Sphericity
```

## 纹理特征参数

纹理特征描述 ROI 内灰度空间分布、相邻关系、连续 run、区域 zone、依赖关系和邻域差异。当前支持以下 PyRadiomics 纹理类别。

### GLCM: Gray Level Co-occurrence Matrix

GLCM 描述一定方向和距离下灰度共现关系，常用于刻画局部灰度对比、均匀性、熵和相关性。

当前可提取 28 个参数：

```text
Autocorrelation, ClusterProminence, ClusterShade, ClusterTendency,
Contrast, Correlation, DifferenceAverage, DifferenceEntropy,
DifferenceVariance, Dissimilarity, Homogeneity1, Homogeneity2, Id,
Idm, Idmn, Idn, Imc1, Imc2, InverseVariance, JointAverage,
JointEnergy, JointEntropy, MCC, MaximumProbability, SumAverage,
SumEntropy, SumSquares, SumVariance
```

### GLRLM: Gray Level Run Length Matrix

GLRLM 描述相同灰度连续出现的 run length，适合刻画条带状、连续性和粗细纹理。

当前可提取 16 个参数：

```text
GrayLevelNonUniformity, GrayLevelNonUniformityNormalized,
GrayLevelVariance, HighGrayLevelRunEmphasis, LongRunEmphasis,
LongRunHighGrayLevelEmphasis, LongRunLowGrayLevelEmphasis,
LowGrayLevelRunEmphasis, RunEntropy, RunLengthNonUniformity,
RunLengthNonUniformityNormalized, RunPercentage, RunVariance,
ShortRunEmphasis, ShortRunHighGrayLevelEmphasis,
ShortRunLowGrayLevelEmphasis
```

### GLSZM: Gray Level Size Zone Matrix

GLSZM 描述相同灰度连通区域 zone 的大小分布，适合刻画区域斑块、粗糙度和均匀性。

当前可提取 16 个参数：

```text
GrayLevelNonUniformity, GrayLevelNonUniformityNormalized,
GrayLevelVariance, HighGrayLevelZoneEmphasis, LargeAreaEmphasis,
LargeAreaHighGrayLevelEmphasis, LargeAreaLowGrayLevelEmphasis,
LowGrayLevelZoneEmphasis, SizeZoneNonUniformity,
SizeZoneNonUniformityNormalized, SmallAreaEmphasis,
SmallAreaHighGrayLevelEmphasis, SmallAreaLowGrayLevelEmphasis,
ZoneEntropy, ZonePercentage, ZoneVariance
```

### GLDM: Gray Level Dependence Matrix

GLDM 描述中心体素与邻域体素之间的灰度依赖关系，适合反映局部依赖、均匀性和复杂度。

当前可提取 16 个参数：

```text
DependenceEntropy, DependenceNonUniformity,
DependenceNonUniformityNormalized, DependencePercentage,
DependenceVariance, GrayLevelNonUniformity,
GrayLevelNonUniformityNormalized, GrayLevelVariance,
HighGrayLevelEmphasis, LargeDependenceEmphasis,
LargeDependenceHighGrayLevelEmphasis,
LargeDependenceLowGrayLevelEmphasis, LowGrayLevelEmphasis,
SmallDependenceEmphasis, SmallDependenceHighGrayLevelEmphasis,
SmallDependenceLowGrayLevelEmphasis
```

### NGTDM: Neighbouring Gray Tone Difference Matrix

NGTDM 描述体素灰度与邻域平均灰度之间的差异，适合刻画粗糙度、复杂度、强度变化和纹理强度。

当前可提取 5 个参数：

```text
Busyness, Coarseness, Complexity, Contrast, Strength
```

## 图像滤波器 / Image Types

Advanced 模式可选择以下图像类型。滤波后会在相应图像上提取所选特征类别。

| 图像类型 | 说明 |
| --- | --- |
| `Original` | 原始影像 |
| `LoG` | Laplacian of Gaussian，sigma 控制强调细纹理或粗纹理 |
| `Wavelet` | 小波分解，支持 LLL、LLH、LHL、LHH、HLL、HLH、HHL、HHH 子带 |
| `Square` | 平方变换 |
| `SquareRoot` | 平方根变换 |
| `Exponential` | 指数变换 |
| `Logarithm` | 对数变换 |
| `Gradient` | 梯度幅值图像 |
| `LBP2D` | 2D Local Binary Pattern |
| `LBP3D` | 3D Local Binary Pattern |

使用 `LBP2D` 或 `LBP3D` 需要 `scikit-image`，已在 `requirements.txt` 中列出。

## 剂量组学功能

如果病例文件夹中存在 RTDOSE，应用会：

1. 读取 RTDOSE 为 SimpleITK 图像；
2. 将所选 ROI mask 与剂量图像对齐；
3. 在剂量分布上提取 PyRadiomics 特征；
4. 生成 Dose Feature Matrix；
5. 支持保存到临时数据库并参与后续建模。

当前版本保留剂量特征提取，不再计算或展示 DTH/OVH。

## 临时数据库与建模

Database & Modeling 页面包含三个标签页。

### Saved Results

- 查看已保存的影像组学/剂量组学结果；
- 生成一例一行的宽表；
- 下载建模用特征表；
- 清空临时数据库。

### Clinical Follow-up

- 上传 CSV、XLSX 或 XLS 表格；
- 选择与 App 中 Case ID 对应的列；
- 保存临床随访信息；
- 与组学特征按 case_id 合并。

### Modeling

支持：

- 选择 Outcome / target column；
- 选择 omics predictors；
- 选择 clinical / follow-up predictors；
- 自动判断 regression 或 classification；
- 自定义 validation fraction、test fraction 和 random seed；
- 数值缺失值填补：median 或 mean；
- 数值标准化；
- 类别变量 one-hot encoding；
- 输出 validation 和 test 评价指标。

## 支持的机器学习方法

### 回归模型

- Linear Regression
- Ridge
- LASSO
- ElasticNet
- Support Vector Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

回归评价指标：

- MAE
- RMSE
- R2

### 分类模型

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

分类评价指标：

- Accuracy
- Balanced Accuracy
- F1
- ROC AUC（二分类且模型支持概率输出时）

### 无监督聚类

- KMeans Clustering
- 输出 cluster assignment；
- 输出 Inertia；
- 样本数允许时输出 Silhouette。

## 统计分析

高级结果页提供以下分析模块：

- Correlation Analysis：Pearson、Spearman、Kendall 相关性；识别高相关特征对和特征簇。
- LASSO Feature Selection：基于 L1 正则化进行特征筛选，可使用交叉验证自动选择 alpha。
- ICC Reliability：用于 test-retest、观察者间/观察者内一致性等重复测量场景。

## 输出文件

常见输出包括：

- `radiomics_features.csv`
- `radiomics_features.xlsx`
- `dose_features.csv`
- `radiomics_modeling_features.csv`
- `model_metrics.csv`
- `lasso_selected_features.csv`
- `icc_results.csv`

临时数据库默认位于：

```text
.radiomics_tmp/radiomics_results.sqlite3
```

该目录已加入 `.gitignore`，不会随代码提交。

## 测试

运行全部测试：

```bash
python3 -m pytest -q
```

语法/导入检查：

```bash
python3 -m compileall app.py src tests
```

## 注意事项

- 影像和 RTSTRUCT 的空间几何必须匹配，否则 ROI mask 可能为空或错位。
- RTDOSE 与 CT 网格不同步时，应用会尝试将 mask 重采样到剂量网格。
- 多影像序列提取时，请确认各序列与 RTSTRUCT 是否来自同一空间参考。
- 建模结果高度依赖样本量、终点定义、数据泄漏控制和外部验证；本工具输出的是探索性建模结果。
- 临时数据库为本地 SQLite 文件，不适合作为多用户生产数据库。

## 技术栈

- Streamlit
- PyRadiomics
- SimpleITK
- pydicom
- pandas / numpy / scipy
- scikit-learn
- scikit-image
- Plotly
- SQLite

## Citation

如果使用本工具发表研究，请同时引用本项目和 PyRadiomics 原始论文。

Radiomics Tools:

```text
Wang, X. Radiomics Tool: Medical Image Feature Extraction & Analysis Platform.
https://github.com/wangxs89/Radiomics-Tools
```

PyRadiomics:

```text
van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A.,
Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C.,
Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System
to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104-e107.
https://doi.org/10.1158/0008-5472.CAN-17-0339
```

## 参考

- PyRadiomics Documentation: https://pyradiomics.readthedocs.io/
- PyRadiomics GitHub: https://github.com/AIM-Harvard/pyradiomics

## License

MIT License
