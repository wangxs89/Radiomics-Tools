"""滤波器模块"""
from typing import Any, Dict, List, Optional


class RadiomicsFilterConfig:
    """PyRadiomics 滤波器配置"""

    AVAILABLE_FILTERS = {
        'original': {
            'name': 'Original',
            'description': '原始影像',
            'params': {}
        },
        'log': {
            'name': 'LoG (Laplacian of Gaussian)',
            'description': '高斯拉普拉斯滤波',
            'params': {'sigma': [1.0, 2.0, 3.0]}
        },
        'wavelet': {
            'name': 'Wavelet',
            'description': '小波变换',
            'params': {'wavelet_type': ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']}
        },
        'gradient': {
            'name': 'Gradient',
            'description': '梯度幅值滤波',
            'params': {}
        },
        'lbp2d': {
            'name': 'LBP 2D',
            'description': '二维局部二值模式滤波',
            'params': {}
        },
        'lbp3d': {
            'name': 'LBP 3D',
            'description': '三维局部二值模式滤波',
            'params': {}
        },
        'square': {
            'name': 'Square',
            'description': '平方滤波',
            'params': {}
        },
        'squareroot': {
            'name': 'Square Root',
            'description': '平方根滤波',
            'params': {}
        },
        'exponential': {
            'name': 'Exponential',
            'description': '指数滤波',
            'params': {}
        },
        'logarithm': {
            'name': 'Logarithm',
            'description': '对数滤波',
            'params': {}
        }
    }

    @staticmethod
    def build_settings(enabled_filters: List[str], log_sigmas: List[float] = None,
                      wavelet_types: List[str] = None,
                      bin_width: Optional[float] = None,
                      resampled_pixel_spacing: Optional[List[float]] = None,
                      force_2d: bool = False,
                      force_2d_dimension: int = 0) -> Dict[str, Any]:
        """构建 PyRadiomics 设置"""
        settings = {
            'enabledImageTypes': [],
            'imageTypeSettings': {},
            'selectedWaveletSubbands': wavelet_types or [],
            'extractorSettings': {}
        }

        if 'original' in enabled_filters:
            settings['enabledImageTypes'].append('Original')
            settings['imageTypeSettings']['Original'] = {}

        if 'log' in enabled_filters:
            sigmas = log_sigmas or [1.0, 2.0, 3.0]
            for sigma in sigmas:
                settings['enabledImageTypes'].append(f'LoG-sigma-{sigma}')
            settings['imageTypeSettings']['LoG'] = {'sigma': sigmas}

        if 'wavelet' in enabled_filters:
            types = wavelet_types or ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            for wt in types:
                settings['enabledImageTypes'].append(f'Wavelet-{wt}')
            settings['imageTypeSettings']['Wavelet'] = {}
            settings['selectedWaveletSubbands'] = types

        if 'gradient' in enabled_filters:
            settings['enabledImageTypes'].append('Gradient')
            settings['imageTypeSettings']['Gradient'] = {}

        if 'lbp2d' in enabled_filters:
            settings['enabledImageTypes'].append('LBP2D')
            settings['imageTypeSettings']['LBP2D'] = {}

        if 'lbp3d' in enabled_filters:
            settings['enabledImageTypes'].append('LBP3D')
            settings['imageTypeSettings']['LBP3D'] = {}

        if 'square' in enabled_filters:
            settings['enabledImageTypes'].append('Square')
            settings['imageTypeSettings']['Square'] = {}

        if 'squareroot' in enabled_filters:
            settings['enabledImageTypes'].append('SquareRoot')
            settings['imageTypeSettings']['SquareRoot'] = {}

        if 'exponential' in enabled_filters:
            settings['enabledImageTypes'].append('Exponential')
            settings['imageTypeSettings']['Exponential'] = {}

        if 'logarithm' in enabled_filters:
            settings['enabledImageTypes'].append('Logarithm')
            settings['imageTypeSettings']['Logarithm'] = {}

        if not settings['imageTypeSettings']:
            settings['enabledImageTypes'].append('Original')
            settings['imageTypeSettings']['Original'] = {}

        if bin_width is not None:
            settings['extractorSettings']['binWidth'] = float(bin_width)

        if resampled_pixel_spacing:
            settings['extractorSettings']['resampledPixelSpacing'] = [
                float(v) for v in resampled_pixel_spacing
            ]

        if force_2d:
            settings['extractorSettings']['force2D'] = True
            settings['extractorSettings']['force2Ddimension'] = int(force_2d_dimension)

        return settings
