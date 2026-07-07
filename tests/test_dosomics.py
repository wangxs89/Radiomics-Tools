from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset

from src.dosomics import image_spacing_to_array_spacing, load_dose_image


def create_test_rtdose(temp_dir: Path) -> Path:
    file_path = temp_dir / "rtdose.dcm"

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7"
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "RTDOSE"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"

    ds.Rows = 2
    ds.Columns = 4
    ds.NumberOfFrames = 3
    ds.PixelSpacing = [1.0, 2.0]  # row, column spacing
    ds.GridFrameOffsetVector = [0.0, 3.0, 6.0]
    ds.ImagePositionPatient = [-10.0, -20.0, -30.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.DoseGridScaling = 0.1
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    pixel_array = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(str(file_path))
    return file_path


def test_load_dose_image_preserves_rtdose_geometry(tmp_path):
    dose_file = create_test_rtdose(tmp_path)
    image = load_dose_image([str(dose_file)])
    array = sitk.GetArrayFromImage(image)

    ds = pydicom.dcmread(str(dose_file))
    expected = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)

    assert image.GetSize() == (4, 2, 3)
    assert image.GetSpacing() == (2.0, 1.0, 3.0)
    assert image.GetOrigin() == (-10.0, -20.0, -30.0)
    assert array.shape == (3, 2, 4)
    assert np.array_equal(array, expected)


def test_image_spacing_to_array_spacing(tmp_path):
    dose_file = create_test_rtdose(tmp_path)
    image = load_dose_image([str(dose_file)])

    assert image_spacing_to_array_spacing(image) == (3.0, 1.0, 2.0)
