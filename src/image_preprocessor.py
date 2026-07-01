"""图像预处理模块"""
import SimpleITK as sitk
import numpy as np
from typing import Optional, List, Tuple


class ImagePreprocessor:
    """图像预处理器"""

    @staticmethod
    def resample_image(image: sitk.Image, new_spacing: List[float],
                       interpolator=sitk.sitkLinear) -> sitk.Image:
        """重采样图像到新体素间距"""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)

        return resampler.Execute(image)

    @staticmethod
    def normalize_image(image: sitk.Image, method: str = "z-score",
                       lower_percentile: float = 1.0,
                       upper_percentile: float = 99.0) -> sitk.Image:
        """归一化图像"""
        array = sitk.GetArrayFromImage(image).astype(np.float64)

        if method == "z-score":
            mean = np.mean(array)
            std = np.std(array)
            if std > 0:
                array = (array - mean) / std

        elif method == "min-max":
            min_val = np.min(array)
            max_val = np.max(array)
            if max_val > min_val:
                array = (array - min_val) / (max_val - min_val)

        elif method == "percentile":
            lower = np.percentile(array, lower_percentile)
            upper = np.percentile(array, upper_percentile)
            array = np.clip(array, lower, upper)
            if upper > lower:
                array = (array - lower) / (upper - lower)

        result = sitk.GetImageFromArray(array)
        result.CopyInformation(image)
        return result

    @staticmethod
    def discretize_image(image: sitk.Image, bin_width: float = 25.0) -> sitk.Image:
        """灰度离散化（固定 bin width）"""
        array = sitk.GetArrayFromImage(image).astype(np.float64)

        min_val = np.min(array)
        discretized = np.floor((array - min_val) / bin_width).astype(np.int32)

        result = sitk.GetImageFromArray(discretized.astype(np.float64))
        result.CopyInformation(image)
        return result
