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
