"""ROI 可视化模块"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from typing import List, Dict, Optional, Tuple
from src.roi_handler import ROI


class ROIVisualizer:
    """ROI 可视化器"""

    COLORS = [
        '#ff4444', '#44ff44', '#4444ff', '#ffff44',
        '#ff44ff', '#44ffff', '#ff8844', '#8844ff',
    ]

    def __init__(self, image: sitk.Image):
        self.image = image
        self.image_array = sitk.GetArrayFromImage(image).astype(np.float64)
        self.spacing = image.GetSpacing()
        self.origin = image.GetOrigin()
        self.direction = image.GetDirection()
        # 4D 支持：(volumes, slices, rows, cols) 或 3D：(slices, rows, cols)
        self.is_4d = self.image_array.ndim == 4
        if self.is_4d:
            self.num_volumes = self.image_array.shape[0]
            self.num_slices = self.image_array.shape[1]
        else:
            self.num_volumes = 1
            self.num_slices = self.image_array.shape[0]

    def _get_slice_2d(self, volume_index: int, slice_index: int) -> np.ndarray:
        """获取 2D 切片数据"""
        if self.is_4d:
            return self.image_array[volume_index, slice_index]
        return self.image_array[slice_index]

    def convert_contour_to_pixels(self, contour_data: List[List[float]], slice_index: int) -> Optional[np.ndarray]:
        if not contour_data:
            return None

        points = []
        # 4D 时 spacing[3] 是 Z 方向间距
        z_spacing = self.spacing[3] if len(self.spacing) > 3 else self.spacing[2]
        z_origin = self.origin[2]
        slice_position = z_origin + slice_index * z_spacing

        for point in contour_data:
            if len(point) >= 3:
                z = point[2]
                if abs(z - slice_position) < z_spacing / 2:
                    x_pixel = (point[0] - self.origin[0]) / self.spacing[0]
                    y_pixel = (point[1] - self.origin[1]) / self.spacing[1]
                    points.append([x_pixel, y_pixel])

        if points:
            return np.array(points)
        return None

    def create_viewer_with_rois(self, rois: List[ROI], selected_rois: List[str] = None,
                                slice_index: int = 0, volume_index: int = 0,
                                window_center: float = None, window_width: float = None):
        slice_data = self._get_slice_2d(volume_index, slice_index)

        if window_center is None:
            window_center = 40
        if window_width is None:
            window_width = 400

        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        im = ax.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax,
                       aspect='equal', interpolation='nearest')

        if selected_rois is None:
            selected_rois = [roi.name for roi in rois]

        legend_handles = []
        for i, roi in enumerate(rois):
            if roi.name in selected_rois:
                color = self.COLORS[i % len(self.COLORS)]
                contour_pixels = self.convert_contour_to_pixels(roi.contour_data, slice_index)

                if contour_pixels is not None and len(contour_pixels) > 0:
                    contour_closed = np.vstack([contour_pixels, contour_pixels[0:1]])
                    line, = ax.plot(contour_closed[:, 0], contour_closed[:, 1],
                                   color=color, linewidth=2, label=roi.name)
                    legend_handles.append(line)

        title = f"Volume {volume_index}, Slice {slice_index}" if self.is_4d else f"Slice {slice_index}"
        ax.set_title(title)
        ax.axis('off')

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right',
                     bbox_to_anchor=(1.3, 1.0), fontsize=9)

        plt.tight_layout()
        return fig

    def get_slice_stats(self, volume_index: int, slice_index: int) -> dict:
        slice_data = self._get_slice_2d(volume_index, slice_index)
        return {
            'min': float(np.nanmin(slice_data)),
            'max': float(np.nanmax(slice_data)),
            'mean': float(np.nanmean(slice_data)),
            'shape': slice_data.shape,
        }

    def create_slice_selector(self, rois: List[ROI], selected_rois: List[str] = None):
        return self.num_slices
