"""影像组学 Web App 主应用"""
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import SimpleITK as sitk

from src.dicom_parser import DICOMParser
from src.roi_handler import ROIHandler
from src.feature_extractor import RadiomicsFeatureExtractor
from src.visualization import ImageViewer
from src.roi_visualizer import ROIVisualizer


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

    # Session state for storing parsed data
    if 'dicom_image' not in st.session_state:
        st.session_state.dicom_image = None
    if 'rois' not in st.session_state:
        st.session_state.rois = None
    if 'roi_handler' not in st.session_state:
        st.session_state.roi_handler = None
    if 'verification_complete' not in st.session_state:
        st.session_state.verification_complete = False

    st.subheader("步骤 1：上传 DICOM 影像")
    dicom_upload = st.file_uploader(
        "上传 DICOM 文件",
        type=['dcm', 'dicom'],
        accept_multiple_files=True,
        key='dicom_uploader'
    )

    dicom_loaded = False
    if dicom_upload:
        st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")

        # Load DICOM series
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, uploaded_file in enumerate(dicom_upload):
                dicom_file = temp_path / f"dicom_{i}.dcm"
                dicom_file.write_bytes(uploaded_file.getvalue())

            parser = DICOMParser()
            series = parser.load_series(str(temp_path))

            if series and series.instances:
                # Convert first instance to SimpleITK image for visualization
                first_instance = series.instances[0]
                ds = first_instance.dataset
                pixel_array = ds.pixel_array.astype(float)

                # Apply rescale slope and intercept if present
                slope = getattr(ds, 'RescaleSlope', 1)
                intercept = getattr(ds, 'Intercept', 0)
                pixel_array = pixel_array * slope + intercept

                # Create SimpleITK image
                sitk_image = sitk.GetImageFromArray(pixel_array)
                spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
                if len(spacing) >= 2:
                    sitk_image.SetSpacing([float(spacing[0]), float(spacing[1])])
                sitk_image.SetOrigin([0.0, 0.0])

                st.session_state.dicom_image = sitk_image
                dicom_loaded = True
                st.success("DICOM 影像加载成功")

    st.subheader("步骤 2：上传 ROI 文件")
    roi_upload = st.file_uploader(
        "上传 RTSTRUCT 文件",
        type=['dcm'],
        accept_multiple_files=False,
        key='roi_uploader'
    )

    roi_loaded = False
    if roi_upload:
        st.success("已上传 RTSTRUCT 文件")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            roi_file = temp_path / "rtstruct.dcm"
            roi_file.write_bytes(roi_upload.getvalue())

            handler = ROIHandler()
            rois = handler.load_rtstruct(str(roi_file))

            if rois:
                st.session_state.rois = rois
                st.session_state.roi_handler = handler
                roi_loaded = True
                st.success(f"ROI 加载成功，共 {len(rois)} 个 ROI")

    # Visualization verification step
    if dicom_loaded and roi_loaded and st.session_state.dicom_image is not None and st.session_state.rois is not None:
        st.markdown("---")
        st.subheader("步骤 3：可视化验证 ROI 位置")

        visualizer = ROIVisualizer(st.session_state.dicom_image)
        rois = st.session_state.rois
        roi_names = handler.get_roi_names(rois)

        # ROI selection
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**选择要显示的 ROI**")
            selected_rois = st.multiselect(
                "ROI 列表",
                roi_names,
                default=roi_names
            )

            # Slice selector
            num_slices = visualizer.image_array.shape[0]
            slice_index = st.slider(
                "切片索引",
                min_value=0,
                max_value=num_slices - 1,
                value=num_slices // 2
            )

        # Display visualization
        with col2:
            fig = visualizer.create_viewer_with_rois(
                rois=rois,
                selected_rois=selected_rois,
                slice_index=slice_index
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confirmation button
        st.markdown("---")
        st.markdown("**确认无误后继续**")
        if st.button("✅ 确认无误，继续提取特征"):
            st.session_state.verification_complete = True
            st.success("验证完成！可以继续特征提取")

    # Show feature extraction if verification is complete
    if st.session_state.verification_complete:
        st.subheader("步骤 4：提取特征")
        rois = st.session_state.rois
        roi_names = handler.get_roi_names(rois)
        selected_rois = st.multiselect(
            "选择要提取特征的 ROI",
            roi_names,
            default=roi_names[:1],
            key='feature_rois'
        )

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
