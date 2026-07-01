"""ROI 可视化模块"""
import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from typing import List, Dict, Optional, Tuple
from src.roi_handler import ROI


class ROIVisualizer:
    """ROI 可视化器"""

    # ROI 颜色映射
    COLORS = [
        '#ff4444',  # 红
        '#44ff44',  # 绿
        '#4444ff',  # 蓝
        '#ffff44',  # 黄
        '#ff44ff',  # 紫
        '#44ffff',  # 青
        '#ff8844',  # 橙
        '#8844ff',  # 紫罗兰
    ]

    def __init__(self, image: sitk.Image):
        """初始化可视化器"""
        self.image = image
        self.image_array = sitk.GetArrayFromImage(image)
        self.spacing = image.GetSpacing()
        self.origin = image.GetOrigin()
        self.direction = image.GetDirection()

    def convert_contour_to_pixels(self, contour_data: List[List[float]], slice_index: int) -> Optional[np.ndarray]:
        """将 RTSTRUCT 轮廓坐标转换为像素坐标"""
        if not contour_data:
            return None

        points = []
        slice_position = self.origin[2] + slice_index * self.spacing[2]

        for point in contour_data:
            if len(point) >= 3:
                z = point[2]
                if abs(z - slice_position) < self.spacing[2] / 2:
                    x_pixel = (point[0] - self.origin[0]) / self.spacing[0]
                    y_pixel = (point[1] - self.origin[1]) / self.spacing[1]
                    points.append([x_pixel, y_pixel])

        if points:
            return np.array(points)
        return None

    def create_viewer_with_rois(self, rois: List[ROI], selected_rois: List[str] = None,
                                slice_index: int = 0, window_center: float = None,
                                window_width: float = None) -> go.Figure:
        """创建带 ROI 叠加的影像查看器

        Args:
            rois: ROI 列表
            selected_rois: 要显示的 ROI 名称列表
            slice_index: 切片索引
            window_center: 窗位（默认 40，软组织窗）
            window_width: 窗宽（默认 400，软组织窗）
        """
        fig = go.Figure()

        slice_data = self.image_array[slice_index]

        # CT 窗宽窗位设置
        if window_center is None:
            window_center = 40  # 软组织窗位
        if window_width is None:
            window_width = 400  # 软组织窗宽

        zmin = window_center - window_width / 2
        zmax = window_center + window_width / 2

        # 显示影像切片
        fig.add_trace(
            go.Heatmap(
                z=slice_data,
                zmin=zmin,
                zmax=zmax,
                colorscale='gray',
                showscale=False,
                hoverinfo='skip',
                xgap=1,
                ygap=1
            )
        )

        # 叠加 ROI 轮廓
        if selected_rois is None:
            selected_rois = [roi.name for roi in rois]

        for i, roi in enumerate(rois):
            if roi.name in selected_rois:
                color = self.COLORS[i % len(self.COLORS)]
                contour_pixels = self.convert_contour_to_pixels(roi.contour_data, slice_index)

                if contour_pixels is not None and len(contour_pixels) > 0:
                    # 闭合轮廓
                    contour_closed = np.vstack([contour_pixels, contour_pixels[0:1]])

                    fig.add_trace(
                        go.Scatter(
                            x=contour_closed[:, 0],
                            y=contour_closed[:, 1],
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=roi.name,
                            hoverinfo='name'
                        )
                    )

        fig.update_layout(
            title=f"影像查看器 - Slice {slice_index}",
            xaxis=dict(visible=False, scaleanchor="y"),
            yaxis=dict(visible=False, autorange="reversed"),
            showlegend=True,
            legend=dict(x=1.05, y=1),
            width=800,
            height=600,
            margin=dict(l=50, r=150, t=50, b=50)
        )

        return fig

    def create_slice_selector(self, rois: List[ROI], selected_rois: List[str] = None):
        """创建带切片选择器的查看器（用于 Streamlit）"""
        num_slices = self.image_array.shape[0]
        return num_slices
