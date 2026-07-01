"""滤波器模块"""
from typing import List, Dict, Any


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
                      wavelet_types: List[str] = None) -> Dict[str, Any]:
        """构建 PyRadiomics 设置"""
        settings = {
            'enabledImageTypes': []
        }

        if 'original' in enabled_filters:
            settings['enabledImageTypes'].append('Original')

        if 'log' in enabled_filters:
            sigmas = log_sigmas or [1.0, 2.0, 3.0]
            for sigma in sigmas:
                settings['enabledImageTypes'].append(f'LoG-sigma-{sigma}')

        if 'wavelet' in enabled_filters:
            types = wavelet_types or ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            for wt in types:
                settings['enabledImageTypes'].append(f'Wavelet-{wt}')

        if 'square' in enabled_filters:
            settings['enabledImageTypes'].append('Square')

        if 'squareroot' in enabled_filters:
            settings['enabledImageTypes'].append('SquareRoot')

        if 'exponential' in enabled_filters:
            settings['enabledImageTypes'].append('Exponential')

        if 'logarithm' in enabled_filters:
            settings['enabledImageTypes'].append('Logarithm')

        return settings
