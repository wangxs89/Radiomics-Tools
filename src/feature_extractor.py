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
