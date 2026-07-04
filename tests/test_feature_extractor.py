import pytest
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from src.filters import RadiomicsFilterConfig
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


def test_feature_extractor_enables_configured_image_types():
    settings = RadiomicsFilterConfig.build_settings(
        enabled_filters=['original', 'log', 'square'],
        log_sigmas=[1.0, 2.0],
    )

    extractor = RadiomicsFeatureExtractor(filter_settings=settings)

    assert extractor.extractor.enabledImagetypes == {
        'Original': {},
        'LoG': {'sigma': [1.0, 2.0]},
        'Square': {},
    }


def test_feature_extractor_filters_wavelet_subbands():
    settings = RadiomicsFilterConfig.build_settings(
        enabled_filters=['wavelet'],
        wavelet_types=['LLH'],
    )
    extractor = RadiomicsFeatureExtractor(filter_settings=settings)

    assert extractor._keep_feature_for_selected_filters('wavelet-LLH_firstorder_Mean')
    assert not extractor._keep_feature_for_selected_filters('wavelet-HHH_firstorder_Mean')
    assert extractor._keep_feature_for_selected_filters('original_firstorder_Mean')
