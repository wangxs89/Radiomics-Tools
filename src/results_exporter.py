"""结果导出模块"""
import pandas as pd
import io
from typing import Dict, Any, Optional


class ResultsExporter:
    """结果导出器"""

    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        """导出为 CSV 字符串"""
        return df.to_csv(index=False, encoding='utf-8-sig')

    @staticmethod
    def to_excel(df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bytes:
        """导出为 Excel 文件（带元数据）"""
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 写入特征数据
            df.to_excel(writer, sheet_name='特征矩阵', index=False)

            # 写入元数据
            if metadata:
                meta_df = pd.DataFrame([metadata])
                meta_df.to_excel(writer, sheet_name='元数据', index=False)

            # 写入特征分类
            feature_categories = ResultsExporter._categorize_features(df)
            cat_df = pd.DataFrame(feature_categories)
            cat_df.to_excel(writer, sheet_name='特征分类', index=False)

        return output.getvalue()

    @staticmethod
    def _categorize_features(df: pd.DataFrame) -> list:
        """对特征进行分类"""
        categories = []
        for col in df.columns:
            if col == 'ROI':
                continue

            if 'shape' in col.lower():
                category = '形状特征'
            elif 'firstorder' in col.lower():
                category = '一阶统计量'
            elif 'glcm' in col.lower():
                category = 'GLCM 纹理'
            elif 'glrlm' in col.lower():
                category = 'GLRLM 纹理'
            elif 'glszm' in col.lower():
                category = 'GLSZM 纹理'
            elif 'gldm' in col.lower():
                category = 'GLDM 纹理'
            elif 'ngtdm' in col.lower():
                category = 'NGTDM 纹理'
            else:
                category = '其他'

            categories.append({
                '特征名称': col,
                '特征类别': category
            })

        return categories

    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
        """获取特征统计摘要"""
        numeric_df = df.select_dtypes(include='number')

        summary = pd.DataFrame({
            '均值': numeric_df.mean(),
            '标准差': numeric_df.std(),
            '中位数': numeric_df.median(),
            '最小值': numeric_df.min(),
            '最大值': numeric_df.max()
        })

        return summary.reset_index().rename(columns={'index': '特征名称'})
