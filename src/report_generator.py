"""可视化报告生成模块"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List


class ReportGenerator:
    """报告生成器"""

    @staticmethod
    def create_feature_distribution(df: pd.DataFrame, feature_name: str) -> go.Figure:
        """创建特征分布直方图"""
        fig = px.histogram(df, x=feature_name, color='ROI',
                          title=f'{feature_name} 分布',
                          nbins=20)
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """创建相关性热力图"""
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu_r',
            zmid=0
        ))

        fig.update_layout(
            title='特征相关性热力图',
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, feature_name: str) -> go.Figure:
        """创建 ROI 对比箱线图"""
        fig = px.box(df, x='ROI', y=feature_name,
                    title=f'{feature_name} - ROI 对比',
                    color='ROI')
        return fig

    @staticmethod
    def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        """创建特征统计摘要表"""
        numeric_df = df.select_dtypes(include='number')

        summary = pd.DataFrame({
            '特征': numeric_df.columns,
            '均值': numeric_df.mean().values,
            '标准差': numeric_df.std().values,
            '中位数': numeric_df.median().values,
            '最小值': numeric_df.min().values,
            '最大值': numeric_df.max().values
        })

        return summary
