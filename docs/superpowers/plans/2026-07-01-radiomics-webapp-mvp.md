# 影像组学 Web App 第一阶段实现计划（MVP）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 实现基础的影像组学特征提取流程，支持 DICOM 影像上传、RTSTRUCT 解析、可视化验证和 PyRadiomics 特征提取

**Architecture:** Streamlit Web App，Python 后端使用 PyRadiomics 提取特征，SimpleITK 处理医学图像，Plotly 实现可视化查看器

**Tech Stack:** Streamlit, PyRadiomics, SimpleITK, pydicom, Plotly, pandas

---

## 文件结构

```
radiomics-webapp/
├── app.py                          # Streamlit 主应用入口
├── requirements.txt                # Python 依赖
├── .streamlit/
│   └── config.toml                 # Streamlit 配置
├── src/
│   ├── __init__.py
│   ├── dicom_parser.py             # DICOM 影像解析
│   ├── roi_handler.py              # RTSTRUCT 解析和 ROI 处理
│   ├── feature_extractor.py        # PyRadiomics 特征提取
│   └── visualization.py            # 影像查看器
├── utils/
│   ├── __init__.py
│   └── file_utils.py               # 文件处理工具
└── tests/
    ├── test_dicom_parser.py
    ├── test_roi_handler.py
    └── test_feature_extractor.py
```

---

## Task 1: 项目初始化和依赖配置

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `src/__init__.py`
- Create: `utils/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
streamlit==1.31.0
pydicom==2.4.4
SimpleITK==2.3.1
pyradiomics==3.1.0
numpy==1.26.3
pandas==2.1.5
plotly==5.18.0
openpyxl==3.1.2
```

- [ ] **Step 2: 创建 Streamlit 配置**

创建 `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

- [ ] **Step 3: 创建包初始化文件**

创建 `src/__init__.py`:

```python
"""影像组学 Web App 核心模块"""
```

创建 `utils/__init__.py`:

```python
"""工具模块"""
```

- [ ] **Step 4: 安装依赖**

```bash
pip install -r requirements.txt
```

Expected: 所有依赖安装成功

- [ ] **Step 5: 提交**

```bash
git init
git add requirements.txt .streamlit/config.toml src/__init__.py utils/__init__.py
git commit -m "chore: initialize project structure and dependencies"
```

---

## Task 2: DICOM 解析模块（TDD）

**Files:**
- Create: `src/dicom_parser.py`
- Test: `tests/test_dicom_parser.py`

- [ ] **Step 1: 编写失败的测试**

创建 `tests/test_dicom_parser.py`:

```python
import pytest
import tempfile
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
from src.dicom_parser import DICOMParser


def create_test_dicom(temp_dir: Path, instance_number: int = 1) -> Path:
    """创建测试用 DICOM 文件"""
    file_path = temp_dir / f"test_{instance_number}.dcm"
    
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = f"1.2.3.{instance_number}"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    
    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number
    
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    pixel_array = np.random.randint(-1000, 1000, (64, 64), dtype=np.int16)
    ds.PixelData = pixel_array.tobytes()
    
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    
    ds.save_as(str(file_path))
    return file_path


def test_dicom_parser_loads_single_file():
    """测试加载单个 DICOM 文件"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dicom_file = create_test_dicom(temp_path, 1)
        
        parser = DICOMParser()
        result = parser.load_dicom(str(dicom_file))
        
        assert result is not None
        assert result.modality == "CT"
        assert result.patient_id == "12345"


def test_dicom_parser_loads_series():
    """测试加载 DICOM 序列"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i in range(1, 4):
            create_test_dicom(temp_path, i)
        
        parser = DICOMParser()
        series = parser.load_series(str(temp_path))
        
        assert series is not None
        assert len(series.instances) == 3
        assert series.modality == "CT"


def test_dicom_parser_get_metadata():
    """测试获取元数据"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dicom_file = create_test_dicom(temp_path, 1)
        
        parser = DICOMParser()
        result = parser.load_dicom(str(dicom_file))
        metadata = parser.get_metadata(result)
        
        assert "modality" in metadata
        assert "patient_id" in metadata
        assert metadata["modality"] == "CT"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_dicom_parser.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'src.dicom_parser'"

- [ ] **Step 3: 实现 DICOM 解析模块**

创建 `src/dicom_parser.py`:

```python
"""DICOM 影像解析模块"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pydicom
import numpy as np
from dataclasses import dataclass


@dataclass
class DICOMInstance:
    """单个 DICOM 实例"""
    file_path: str
    dataset: pydicom.Dataset
    instance_number: int
    position: List[float]


@dataclass
class DICOMSeries:
    """DICOM 序列"""
    series_uid: str
    modality: str
    patient_id: str
    instances: List[DICOMInstance]
    
    def __init__(self, series_uid: str, modality: str, patient_id: str):
        self.series_uid = series_uid
        self.modality = modality
        self.patient_id = patient_id
        self.instances = []


class DICOMParser:
    """DICOM 影像解析器"""
    
    def load_dicom(self, file_path: str) -> Optional[DICOMInstance]:
        """加载单个 DICOM 文件"""
        try:
            dataset = pydicom.dcmread(file_path)
            
            instance = DICOMInstance(
                file_path=file_path,
                dataset=dataset,
                instance_number=getattr(dataset, 'InstanceNumber', 1),
                position=getattr(dataset, 'ImagePositionPatient', [0.0, 0.0, 0.0])
            )
            
            return instance
        except Exception as e:
            print(f"加载 DICOM 文件失败: {e}")
            return None
    
    def load_series(self, directory: str) -> Optional[DICOMSeries]:
        """加载 DICOM 序列（从目录）"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"目录不存在: {directory}")
            return None
        
        dicom_files = []
        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in ['.dcm', '.dicom', '']:
                try:
                    dataset = pydicom.dcmread(str(file_path))
                    dicom_files.append(dataset)
                except Exception:
                    continue
        
        if not dicom_files:
            print("未找到 DICOM 文件")
            return None
        
        first_ds = dicom_files[0]
        series = DICOMSeries(
            series_uid=getattr(first_ds, 'SeriesInstanceUID', 'unknown'),
            modality=getattr(first_ds, 'Modality', 'unknown'),
            patient_id=getattr(first_ds, 'PatientID', 'unknown')
        )
        
        for ds in dicom_files:
            instance = DICOMInstance(
                file_path=str(ds.filename),
                dataset=ds,
                instance_number=getattr(ds, 'InstanceNumber', 1),
                position=getattr(ds, 'ImagePositionPatient', [0.0, 0.0, 0.0])
            )
            series.instances.append(instance)
        
        series.instances.sort(key=lambda x: x.instance_number)
        
        return series
    
    def get_metadata(self, instance: DICOMInstance) -> Dict[str, Any]:
        """获取 DICOM 元数据"""
        ds = instance.dataset
        
        metadata = {
            'modality': getattr(ds, 'Modality', 'unknown'),
            'patient_id': getattr(ds, 'PatientID', 'unknown'),
            'patient_name': str(getattr(ds, 'PatientName', 'unknown')),
            'series_description': getattr(ds, 'SeriesDescription', ''),
            'rows': getattr(ds, 'Rows', 0),
            'columns': getattr(ds, 'Columns', 0),
            'pixel_spacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
            'slice_thickness': getattr(ds, 'SliceThickness', 1.0),
        }
        
        return metadata
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_dicom_parser.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/dicom_parser.py tests/test_dicom_parser.py
git commit -m "feat: implement DICOM parser module with TDD"
```

---

## Task 3: ROI 处理模块（TDD）

**Files:**
- Create: `src/roi_handler.py`
- Test: `tests/test_roi_handler.py`

- [ ] **Step 1: 编写失败的测试**

创建 `tests/test_roi_handler.py`:

```python
import pytest
import tempfile
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset, FileDataset
from src.roi_handler import ROIHandler, ROI


def create_test_rtstruct(temp_dir: Path) -> Path:
    """创建测试用 RTSTRUCT 文件"""
    file_path = temp_dir / "rtstruct.dcm"
    
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    
    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.Modality = "RTSTRUCT"
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    
    roi_sequence = []
    
    for i, name in enumerate(["GTV", "CTV", "Heart"], start=1):
        roi_obs = Dataset()
        roi_obs.ObservationNumber = i
        roi_obs.ROINumber = i
        roi_obs.ReferencedROINumber = i
        roi_obs.ROIObservationLabel = name
        
        roi_struct = Dataset()
        roi_struct.ROINumber = i
        roi_struct.ROIName = name
        roi_struct.ROIGenerationAlgorithm = "MANUAL"
        
        contour_seq = Dataset()
        contour_seq.ContourNumber = 1
        contour_seq.ContourGeometricType = "CLOSED_PLANAR"
        contour_seq.NumberOfContourPoints = 4
        contour_seq.ContourData = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        
        roi_struct.ContourSequence = [contour_seq]
        roi_sequence.append(roi_struct)
    
    ds.StructureSetROISequence = roi_sequence
    ds.ROIContourSequence = []
    
    ds.save_as(str(file_path))
    return file_path


def test_roi_handler_loads_rtstruct():
    """测试加载 RTSTRUCT"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rtstruct_file = create_test_rtstruct(temp_path)
        
        handler = ROIHandler()
        rois = handler.load_rtstruct(str(rtstruct_file))
        
        assert rois is not None
        assert len(rois) == 3
        assert rois[0].name == "GTV"
        assert rois[1].name == "CTV"
        assert rois[2].name == "Heart"


def test_roi_handler_get_roi_by_name():
    """测试按名称获取 ROI"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rtstruct_file = create_test_rtstruct(temp_path)
        
        handler = ROIHandler()
        rois = handler.load_rtstruct(str(rtstruct_file))
        
        gtv = handler.get_roi_by_name(rois, "GTV")
        assert gtv is not None
        assert gtv.name == "GTV"
        
        unknown = handler.get_roi_by_name(rois, "Unknown")
        assert unknown is None


def test_roi_handler_get_roi_names():
    """测试获取所有 ROI 名称"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rtstruct_file = create_test_rtstruct(temp_path)
        
        handler = ROIHandler()
        rois = handler.load_rtstruct(str(rtstruct_file))
        
        names = handler.get_roi_names(rois)
        assert names == ["GTV", "CTV", "Heart"]
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_roi_handler.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'src.roi_handler'"

- [ ] **Step 3: 实现 ROI 处理模块**

创建 `src/roi_handler.py`:

```python
"""ROI 处理模块"""
from typing import List, Optional
from dataclasses import dataclass
import pydicom


@dataclass
class ROI:
    """感兴趣区域"""
    number: int
    name: str
    contour_data: List[List[float]]


class ROIHandler:
    """ROI 处理器"""
    
    def load_rtstruct(self, file_path: str) -> Optional[List[ROI]]:
        """加载 RTSTRUCT 文件"""
        try:
            dataset = pydicom.dcmread(file_path)
            
            if dataset.Modality != "RTSTRUCT":
                print("不是 RTSTRUCT 文件")
                return None
            
            rois = []
            
            if hasattr(dataset, 'StructureSetROISequence'):
                for roi_seq in dataset.StructureSetROISequence:
                    roi_number = roi_seq.ROINumber
                    roi_name = roi_seq.ROIName
                    
                    contour_data = []
                    
                    if hasattr(dataset, 'ROIContourSequence'):
                        for contour_seq in dataset.ROIContourSequence:
                            if hasattr(contour_seq, 'ReferencedROINumber'):
                                if contour_seq.ReferencedROINumber == roi_number:
                                    if hasattr(contour_seq, 'ContourSequence'):
                                        for contour in contour_seq.ContourSequence:
                                            if hasattr(contour, 'ContourData'):
                                                points = contour.ContourData
                                                for i in range(0, len(points), 3):
                                                    if i + 2 < len(points):
                                                        contour_data.append([
                                                            float(points[i]),
                                                            float(points[i+1]),
                                                            float(points[i+2])
                                                        ])
                    
                    roi = ROI(
                        number=roi_number,
                        name=roi_name,
                        contour_data=contour_data
                    )
                    rois.append(roi)
            
            return rois
        except Exception as e:
            print(f"加载 RTSTRUCT 失败: {e}")
            return None
    
    def get_roi_by_name(self, rois: List[ROI], name: str) -> Optional[ROI]:
        """按名称获取 ROI"""
        for roi in rois:
            if roi.name == name:
                return roi
        return None
    
    def get_roi_names(self, rois: List[ROI]) -> List[str]:
        """获取所有 ROI 名称"""
        return [roi.name for roi in rois]
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_roi_handler.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/roi_handler.py tests/test_roi_handler.py
git commit -m "feat: implement ROI handler module with TDD"
```

---

## Task 4: 特征提取模块（TDD）

**Files:**
- Create: `src/feature_extractor.py`
- Test: `tests/test_feature_extractor.py`

- [ ] **Step 1: 编写失败的测试**

创建 `tests/test_feature_extractor.py`:

```python
import pytest
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from src.feature_extractor import RadiomicsFeatureExtractor


def create_test_image(size=(10, 10, 10)) -> sitk.Image:
    """创建测试用 SimpleITK 图像"""
    image = sitk.GetImageFromArray(np.random.rand(*size))
    image.SetSpacing([1.0, 1.0, 1.0])
    return image


def create_test_mask(size=(10, 10, 10)) -> sitk.Image:
    """创建测试用掩模"""
    mask_array = np.zeros(size, dtype=np.uint8)
    mask_array[3:7, 3:7, 3:7] = 1
    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing([1.0, 1.0, 1.0])
    return mask


def test_feature_extractor_extracts_features():
    """测试特征提取"""
    image = create_test_image()
    mask = create_test_mask()
    
    extractor = RadiomicsFeatureExtractor()
    features = extractor.extract_features(image, mask)
    
    assert features is not None
    assert len(features) > 0
    assert 'original_shape_Volume' in features or 'original_firstorder_Mean' in features


def test_feature_extractor_returns_dataframe():
    """测试返回 DataFrame"""
    image = create_test_image()
    mask = create_test_mask()
    
    extractor = RadiomicsFeatureExtractor()
    df = extractor.extract_to_dataframe(image, mask, roi_name="TestROI")
    
    assert df is not None
    assert 'ROI' in df.columns
    assert df['ROI'].iloc[0] == "TestROI"
    assert len(df) == 1


def test_feature_extractor_handles_empty_mask():
    """测试处理空掩模"""
    image = create_test_image()
    mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing([1.0, 1.0, 1.0])
    
    extractor = RadiomicsFeatureExtractor()
    features = extractor.extract_features(image, mask)
    
    assert features is None or len(features) == 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_feature_extractor.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'src.feature_extractor'"

- [ ] **Step 3: 实现特征提取模块**

创建 `src/feature_extractor.py`:

```python
"""影像组学特征提取模块"""
from typing import Dict, Optional
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor


class RadiomicsFeatureExtractor:
    """PyRadiomics 特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        self.extractor.enableFeatureClassByName('shape')
        self.extractor.enableFeatureClassByName('firstorder')
        self.extractor.enableFeatureClassByName('glcm')
        self.extractor.enableFeatureClassByName('glrlm')
        self.extractor.enableFeatureClassByName('glszm')
        self.extractor.enableFeatureClassByName('gldm')
        self.extractor.enableFeatureClassByName('ngtdm')
    
    def extract_features(self, image: sitk.Image, mask: sitk.Image) -> Optional[Dict[str, float]]:
        """提取影像组学特征"""
        try:
            mask_array = sitk.GetArrayFromImage(mask)
            
            if np.sum(mask_array) == 0:
                print("掩模为空")
                return None
            
            result = self.extractor.execute(image, mask)
            
            features = {}
            for key, value in result.items():
                if not key.startswith('diagnostics_'):
                    features[key] = float(value)
            
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def extract_to_dataframe(self, image: sitk.Image, mask: sitk.Image, 
                            roi_name: str = "ROI") -> Optional[pd.DataFrame]:
        """提取特征并转换为 DataFrame"""
        features = self.extract_features(image, mask)
        
        if features is None:
            return None
        
        df = pd.DataFrame([features])
        df.insert(0, 'ROI', roi_name)
        
        return df
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_feature_extractor.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/feature_extractor.py tests/test_feature_extractor.py
git commit -m "feat: implement radiomics feature extractor with TDD"
```

---

## Task 5: 可视化模块

**Files:**
- Create: `src/visualization.py`

- [ ] **Step 1: 实现影像查看器**

创建 `src/visualization.py`:

```python
"""可视化模块"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import SimpleITK as sitk


class ImageViewer:
    """医学影像查看器"""
    
    def __init__(self, image: sitk.Image):
        """初始化查看器"""
        self.image = image
        self.image_array = sitk.GetArrayFromImage(image)
    
    def create_slice_viewer(self, roi_contours=None, title="影像查看器"):
        """创建切片浏览器"""
        num_slices = self.image_array.shape[0]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=self.image_array[0],
                colorscale='gray',
                showscale=False
            )
        )
        
        if roi_contours:
            for contour in roi_contours:
                fig.add_trace(
                    go.Scatter(
                        x=contour[:, 0],
                        y=contour[:, 1],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='ROI'
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True
                    }],
                    'label': '▶ Play',
                    'method': 'animate'
                }, {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': '⏸ Pause',
                    'method': 'animate'
                }],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 12},
                    'prefix': 'Slice:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'args': [[i], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(num_slices)]
            }]
        )
        
        frames = []
        for i in range(num_slices):
            frame_data = [go.Heatmap(z=self.image_array[i], colorscale='gray', showscale=False)]
            
            if roi_contours:
                for contour in roi_contours:
                    frame_data.append(
                        go.Scatter(
                            x=contour[:, 0],
                            y=contour[:, 1],
                            mode='lines',
                            line=dict(color='red', width=2)
                        )
                    )
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        fig.frames = frames
        
        return fig
```

- [ ] **Step 2: 提交**

```bash
git add src/visualization.py
git commit -m "feat: implement image visualization module"
```

---

## Task 6: Streamlit 主应用

**Files:**
- Create: `app.py`

- [ ] **Step 1: 创建主应用**

创建 `app.py`:

```python
"""影像组学 Web App 主应用"""
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

from src.dicom_parser import DICOMParser
from src.roi_handler import ROIHandler
from src.feature_extractor import RadiomicsFeatureExtractor
from src.visualization import ImageViewer


st.set_page_config(
    page_title="影像组学 Web App",
    page_icon="🏥",
    layout="wide"
)


def main():
    st.title("🏥 影像组学 Web App")
    st.markdown("---")
    
    st.sidebar.header("模式选择")
    mode = st.sidebar.radio("选择操作模式", ["初级模式", "高级模式"])
    
    if mode == "初级模式":
        beginner_mode()
    else:
        advanced_mode()


def beginner_mode():
    """初级模式：一键提取"""
    st.header("🟢 初级模式")
    st.markdown("简单三步，快速提取影像组学特征")
    
    st.subheader("步骤 1：上传 DICOM 影像")
    dicom_upload = st.file_uploader(
        "上传 DICOM 文件",
        type=['dcm', 'dicom'],
        accept_multiple_files=True
    )
    
    if dicom_upload:
        st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")
    
    st.subheader("步骤 2：上传 ROI 文件")
    roi_upload = st.file_uploader(
        "上传 RTSTRUCT 文件",
        type=['dcm'],
        accept_multiple_files=False
    )
    
    if roi_upload:
        st.success("已上传 RTSTRUCT 文件")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            roi_file = temp_path / "rtstruct.dcm"
            roi_file.write_bytes(roi_upload.getvalue())
            
            handler = ROIHandler()
            rois = handler.load_rtstruct(str(roi_file))
            
            if rois:
                st.subheader("步骤 3：选择 ROI")
                roi_names = handler.get_roi_names(rois)
                selected_rois = st.multiselect("选择要提取特征的 ROI", roi_names, default=roi_names[:1])
                
                if st.button("🚀 一键提取特征"):
                    with st.spinner("正在提取特征..."):
                        st.info("特征提取功能开发中...")
                        st.write("选中的 ROI:", selected_rois)


def advanced_mode():
    """高级模式：精细控制"""
    st.header("🟡 高级模式")
    st.markdown("详细参数控制，满足高级需求")
    
    st.info("高级模式功能开发中...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试运行**

```bash
streamlit run app.py
```

Expected: 浏览器打开应用，显示初级模式界面

- [ ] **Step 3: 提交**

```bash
git add app.py
git commit -m "feat: implement Streamlit main application"
```

---

## Task 7: 集成测试和文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README**

创建 `README.md`:

```markdown
# 影像组学 Web App

基于 Streamlit 的影像组学特征提取 Web 应用

## 功能特点

- 🟢 **初级模式**：一键上传，自动提取
- 🟡 **高级模式**：精细控制参数
- 📊 **可视化验证**：检查 ROI 位置
- 📈 **统计分析**：ICC、LASSO、相关性分析

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 推送代码到 GitHub
2. 访问 https://share.streamlit.io
3. 选择仓库和主文件（app.py）
4. 点击 Deploy

## 技术栈

- Streamlit
- PyRadiomics
- SimpleITK
- pydicom
- Plotly

## 许可证

MIT License
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## 完成检查清单

- [ ] 所有测试通过
- [ ] 应用可以运行
- [ ] 可以上传 DICOM 文件
- [ ] 可以上传 RTSTRUCT 文件
- [ ] 可以查看 ROI 列表
- [ ] 代码已提交

---

**下一步：** 第二阶段将添加可视化验证界面、图像预处理选项和完整的结果导出功能
