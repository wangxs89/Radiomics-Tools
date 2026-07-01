"""可视化模块"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import SimpleITK as sitk


class ImageViewer:
    """医学影像查看器"""

    def __init__(self, image: sitk.Image):
        """初始化查看器"""
        self.image = image
        self.image_array = sitk.GetArrayFromImage(image)

    def create_slice_viewer(self, roi_contours=None, title="影像查看器"):
        """创建切片浏览器"""
        num_slices = self.image_array.shape[0]

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=self.image_array[0],
                colorscale='gray',
                showscale=False
            )
        )

        if roi_contours:
            for contour in roi_contours:
                fig.add_trace(
                    go.Scatter(
                        x=contour[:, 0],
                        y=contour[:, 1],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='ROI'
                    )
                )

        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True
                    }],
                    'label': '▶ Play',
                    'method': 'animate'
                }, {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': '⏸ Pause',
                    'method': 'animate'
                }],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 12},
                    'prefix': 'Slice:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'args': [[i], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(num_slices)]
            }]
        )

        frames = []
        for i in range(num_slices):
            frame_data = [go.Heatmap(z=self.image_array[i], colorscale='gray', showscale=False)]

            if roi_contours:
                for contour in roi_contours:
                    frame_data.append(
                        go.Scatter(
                            x=contour[:, 0],
                            y=contour[:, 1],
                            mode='lines',
                            line=dict(color='red', width=2)
                        )
                    )

            frames.append(go.Frame(data=frame_data, name=str(i)))

        fig.frames = frames

        return fig
