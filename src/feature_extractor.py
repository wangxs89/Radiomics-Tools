"""影像组学特征提取模块"""
from typing import Dict, Optional, List
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor


class RadiomicsFeatureExtractor:
    """PyRadiomics 特征提取器"""

    def __init__(self, feature_classes: Dict[str, bool] = None, filter_settings: Dict = None):
        """初始化特征提取器

        Args:
            feature_classes: 字典，指定要启用的特征类别
            filter_settings: 字典，包含滤波器设置
        """
        # 默认启用所有特征类别
        if feature_classes is None:
            feature_classes = {
                'shape': True,
                'firstorder': True,
                'glcm': True,
                'gldm': True,
                'glrlm': True,
                'glszm': True,
                'ngtdm': True
            }

        self.feature_classes = feature_classes
        self.filter_settings = filter_settings or {'enabledImageTypes': ['Original']}

        # 创建特征提取器
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

        # 禁用所有特征类别
        self.extractor.disableAllFeatures()

        # 启用指定的特征类别
        for feature_class, enabled in feature_classes.items():
            if enabled:
                self.extractor.enableFeatureClassByName(feature_class)

        # 配置滤波器
        if 'enabledImageTypes' in self.filter_settings:
            self.extractor._settings['enabledImageTypes'] = self.filter_settings['enabledImageTypes']

    def convert_dicom_series_to_sitk(self, dicom_series) -> sitk.Image:
        """将 DICOM 序列转换为 SimpleITK Image"""
        import os

        # 获取所有 DICOM 文件路径
        file_paths = [inst.file_path for inst in dicom_series.instances]

        # 使用 SimpleITK 读取 DICOM 序列
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        image = reader.Execute()

        return image

    def convert_roi_to_mask(self, roi, dicom_series, image: sitk.Image) -> sitk.Image:
        """将 RTSTRUCT ROI 转换为二值掩模"""
        # 获取影像尺寸
        size = image.GetSize()

        # 创建空掩模
        mask_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        # 将轮廓点转换为像素坐标并填充
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        # 按 Z 坐标分组轮廓点
        z_slices = {}
        for point in roi.contour_data:
            if len(point) >= 3:
                z = point[2]
                z_idx = int(round((z - origin[2]) / spacing[2]))
                if 0 <= z_idx < size[2]:
                    if z_idx not in z_slices:
                        z_slices[z_idx] = []
                    x_pixel = int(round((point[0] - origin[0]) / spacing[0]))
                    y_pixel = int(round((point[1] - origin[1]) / spacing[1]))
                    z_slices[z_idx].append([x_pixel, y_pixel])

        # 对每个切片填充轮廓内部
        for z_idx, points in z_slices.items():
            if len(points) >= 3:
                points = np.array(points)
                # 使用边界框内的点
                min_x, min_y = points.min(axis=0)
                max_x, max_y = points.max(axis=0)

                # 简单的扫描线填充
                for y in range(max(0, min_y), min(size[1], max_y + 1)):
                    # 找到与 y 相交的边
                    intersections = []
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        if p1[1] <= y < p2[1] or p2[1] <= y < p1[1]:
                            if p2[1] != p1[1]:
                                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                                intersections.append(int(round(x)))

                    intersections.sort()
                    # 填充交点之间
                    for i in range(0, len(intersections) - 1, 2):
                        if i + 1 < len(intersections):
                            x_start = max(0, intersections[i])
                            x_end = min(size[0], intersections[i + 1])
                            mask_array[z_idx, y, x_start:x_end] = 1

        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)

        return mask

    def extract_features_for_rois(self, image: sitk.Image, masks_dict: Dict[str, sitk.Image],
                                   feature_classes: Dict[str, bool] = None,
                                   filter_settings: Dict = None) -> pd.DataFrame:
        """为多个 ROI 提取特征

        Args:
            image: SimpleITK 图像
            masks_dict: ROI 名称到掩模的字典
            feature_classes: 特征类别配置
            filter_settings: 滤波器设置
        """
        # 如果提供了新的配置，重新初始化提取器
        if feature_classes is not None or filter_settings is not None:
            self.__init__(feature_classes=feature_classes, filter_settings=filter_settings)

        all_features = []

        for roi_name, mask in masks_dict.items():
            features = self.extract_features(image, mask)
            if features:
                features['ROI'] = roi_name
                all_features.append(features)

        if all_features:
            df = pd.DataFrame(all_features)
            # 重新排列列，ROI 在第一列
            cols = ['ROI'] + [c for c in df.columns if c != 'ROI']
            return df[cols]

        return pd.DataFrame()

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
