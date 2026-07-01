"""DICOM 影像解析模块"""
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

    @property
    def modality(self) -> str:
        return getattr(self.dataset, 'Modality', 'unknown')

    @property
    def patient_id(self) -> str:
        return getattr(self.dataset, 'PatientID', 'unknown')


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
