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
