"""影像组学 Web App 主应用"""
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

from src.dicom_parser import DICOMParser
from src.roi_handler import ROIHandler
from src.feature_extractor import RadiomicsFeatureExtractor
from src.visualization import ImageViewer


st.set_page_config(
    page_title="影像组学 Web App",
    page_icon="🏥",
    layout="wide"
)


def main():
    st.title("🏥 影像组学 Web App")
    st.markdown("---")

    st.sidebar.header("模式选择")
    mode = st.sidebar.radio("选择操作模式", ["初级模式", "高级模式"])

    if mode == "初级模式":
        beginner_mode()
    else:
        advanced_mode()


def beginner_mode():
    """初级模式：一键提取"""
    st.header("🟢 初级模式")
    st.markdown("简单三步，快速提取影像组学特征")

    st.subheader("步骤 1：上传 DICOM 影像")
    dicom_upload = st.file_uploader(
        "上传 DICOM 文件",
        type=['dcm', 'dicom'],
        accept_multiple_files=True
    )

    if dicom_upload:
        st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")

    st.subheader("步骤 2：上传 ROI 文件")
    roi_upload = st.file_uploader(
        "上传 RTSTRUCT 文件",
        type=['dcm'],
        accept_multiple_files=False
    )

    if roi_upload:
        st.success("已上传 RTSTRUCT 文件")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            roi_file = temp_path / "rtstruct.dcm"
            roi_file.write_bytes(roi_upload.getvalue())

            handler = ROIHandler()
            rois = handler.load_rtstruct(str(roi_file))

            if rois:
                st.subheader("步骤 3：选择 ROI")
                roi_names = handler.get_roi_names(rois)
                selected_rois = st.multiselect("选择要提取特征的 ROI", roi_names, default=roi_names[:1])

                if st.button("🚀 一键提取特征"):
                    with st.spinner("正在提取特征..."):
                        st.info("特征提取功能开发中...")
                        st.write("选中的 ROI:", selected_rois)


def advanced_mode():
    """高级模式：精细控制"""
    st.header("🟡 高级模式")
    st.markdown("详细参数控制，满足高级需求")

    st.info("高级模式功能开发中...")


if __name__ == "__main__":
    main()
