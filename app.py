"""影像组学 Web App 主应用"""
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import SimpleITK as sitk

from src.dicom_parser import DICOMParser, DICOMSeries
from src.roi_handler import ROIHandler
from src.feature_extractor import RadiomicsFeatureExtractor
from src.visualization import ImageViewer
from src.roi_visualizer import ROIVisualizer
from src.results_exporter import ResultsExporter
from src.image_preprocessor import ImagePreprocessor


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
    if 'dicom_series' not in st.session_state:
        st.session_state.dicom_series = None
    if 'rois' not in st.session_state:
        st.session_state.rois = None
    if 'roi_handler' not in st.session_state:
        st.session_state.roi_handler = None
    if 'verification_complete' not in st.session_state:
        st.session_state.verification_complete = False
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = None

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
                # Store the DICOM series for later use
                st.session_state.dicom_series = series

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

        # Initialize feature extractor
        if st.session_state.feature_extractor is None:
            st.session_state.feature_extractor = RadiomicsFeatureExtractor()

        rois = st.session_state.rois
        roi_names = handler.get_roi_names(rois)
        selected_rois = st.multiselect(
            "选择要提特征的 ROI",
            roi_names,
            default=roi_names[:1],
            key='feature_rois'
        )

        if st.button("🚀 一键提取特征"):
            with st.spinner("正在提取特征..."):
                try:
                    # Get the feature extractor
                    feature_extractor = st.session_state.feature_extractor
                    dicom_series = st.session_state.dicom_series
                    dicom_image = st.session_state.dicom_image

                    # Convert DICOM series to SimpleITK image using the feature extractor
                    if dicom_series is not None:
                        sitk_image = feature_extractor.convert_dicom_series_to_sitk(dicom_series)
                    else:
                        sitk_image = dicom_image

                    # Process selected ROIs
                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
                            with st.spinner(f"正在处理 ROI: {roi.name}"):
                                mask = feature_extractor.convert_roi_to_mask(roi, dicom_series, sitk_image)
                                masks_dict[roi.name] = mask

                    # Extract features for all ROIs
                    if masks_dict:
                        df_features = feature_extractor.extract_features_for_rois(sitk_image, masks_dict)

                        if not df_features.empty:
                            st.success(f"特征提取完成！共提取了 {len(df_features)} 个 ROI 的特征")

                            # Display features
                            st.subheader("特征矩阵")
                            st.dataframe(df_features, use_container_width=True)

                            # Download buttons using ResultsExporter
                            exporter = ResultsExporter()

                            # CSV download
                            csv = exporter.to_csv(df_features)
                            st.download_button(
                                label="📥 下载特征矩阵 (CSV)",
                                data=csv,
                                file_name="radiomics_features.csv",
                                mime="text/csv"
                            )

                            # Excel download with metadata
                            metadata = {
                                '影像名称': getattr(dicom_series, 'series_description', ''),
                                '患者ID': getattr(dicom_series.instances[0].dataset, 'PatientID', '') if dicom_series.instances else '',
                                'ROI数量': len(masks_dict),
                                '特征总数': len(df_features.columns) - 1  # excluding ROI column
                            }
                            excel_bytes = exporter.to_excel(df_features, metadata=metadata)
                            st.download_button(
                                label="📥 下载特征矩阵 (Excel)",
                                data=excel_bytes,
                                file_name="radiomics_features.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            # Display summary statistics
                            st.subheader("特征统计摘要")
                            summary_stats = exporter.get_summary_stats(df_features)
                            st.dataframe(summary_stats, use_container_width=True)
                        else:
                            st.warning("未能提取到任何特征")

                except Exception as e:
                    st.error(f"特征提取失败: {e}")
                    st.exception(e)


def advanced_mode():
    """高级模式：精细控制"""
    st.header("🟡 高级模式")
    st.markdown("详细参数控制，满足高级需求")

    # Session state for advanced mode
    if 'adv_dicom_image' not in st.session_state:
        st.session_state.adv_dicom_image = None
    if 'adv_dicom_series' not in st.session_state:
        st.session_state.adv_dicom_series = None
    if 'adv_rois' not in st.session_state:
        st.session_state.adv_rois = None
    if 'adv_roi_handler' not in st.session_state:
        st.session_state.adv_roi_handler = None
    if 'adv_verification_complete' not in st.session_state:
        st.session_state.adv_verification_complete = False
    if 'adv_feature_extractor' not in st.session_state:
        st.session_state.adv_feature_extractor = None
    if 'adv_preprocessed_image' not in st.session_state:
        st.session_state.adv_preprocessed_image = None

    st.subheader("步骤 1：上传 DICOM 影像")
    dicom_upload = st.file_uploader(
        "上传 DICOM 文件",
        type=['dcm', 'dicom'],
        accept_multiple_files=True,
        key='adv_dicom_uploader'
    )

    dicom_loaded = False
    if dicom_upload:
        st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, uploaded_file in enumerate(dicom_upload):
                dicom_file = temp_path / f"dicom_{i}.dcm"
                dicom_file.write_bytes(uploaded_file.getvalue())

            parser = DICOMParser()
            series = parser.load_series(str(temp_path))

            if series and series.instances:
                st.session_state.adv_dicom_series = series

                first_instance = series.instances[0]
                ds = first_instance.dataset
                pixel_array = ds.pixel_array.astype(float)

                slope = getattr(ds, 'RescaleSlope', 1)
                intercept = getattr(ds, 'Intercept', 0)
                pixel_array = pixel_array * slope + intercept

                sitk_image = sitk.GetImageFromArray(pixel_array)
                spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
                if len(spacing) >= 2:
                    sitk_image.SetSpacing([float(spacing[0]), float(spacing[1])])
                sitk_image.SetOrigin([0.0, 0.0])

                st.session_state.adv_dicom_image = sitk_image
                dicom_loaded = True
                st.success("DICOM 影像加载成功")

    st.subheader("步骤 2：上传 ROI 文件")
    roi_upload = st.file_uploader(
        "上传 RTSTRUCT 文件",
        type=['dcm'],
        accept_multiple_files=False,
        key='adv_roi_uploader'
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
                st.session_state.adv_rois = rois
                st.session_state.adv_roi_handler = handler
                roi_loaded = True
                st.success(f"ROI 加载成功，共 {len(rois)} 个 ROI")

    # Preprocessing options
    st.markdown("---")
    st.subheader("步骤 3：图像预处理选项")

    preprocess_enabled = st.checkbox("启用图像预处理", value=False)

    preprocessor = ImagePreprocessor()
    original_image = st.session_state.adv_dicom_image
    processed_image = original_image

    if preprocess_enabled and original_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**重采样**")
            enable_resample = st.checkbox("启用重采样", value=False)
            if enable_resample:
                new_spacing_x = st.number_input("X 方向间距 (mm)", value=1.0, step=0.1)
                new_spacing_y = st.number_input("Y 方向间距 (mm)", value=1.0, step=0.1)
                new_spacing_z = st.number_input("Z 方向间距 (mm)", value=1.0, step=0.1)

                if st.button("应用重采样"):
                    processed_image = preprocessor.resample_image(
                        processed_image,
                        [new_spacing_x, new_spacing_y, new_spacing_z]
                    )
                    st.success("重采样完成")

        with col2:
            st.markdown("**归一化**")
            normalize_method = st.selectbox(
                "归一化方法",
                ["无", "z-score", "min-max", "percentile"]
            )
            if normalize_method != "无":
                if st.button("应用归一化"):
                    processed_image = preprocessor.normalize_image(processed_image, method=normalize_method)
                    st.success(f"{normalize_method} 归一化完成")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**灰度离散化**")
            enable_discretize = st.checkbox("启用离散化", value=False)
            if enable_discretize:
                bin_width = st.number_input("Bin 宽度", value=25.0, step=1.0)
                if st.button("应用离散化"):
                    processed_image = preprocessor.discretize_image(processed_image, bin_width=bin_width)
                    st.success("离散化完成")

        if processed_image is not original_image:
            st.session_state.adv_preprocessed_image = processed_image
            st.info("预处理后的图像已保存，将用于特征提取")

    # Visualization verification step
    image_for_viz = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image
    if dicom_loaded and roi_loaded and image_for_viz is not None and st.session_state.adv_rois is not None:
        st.markdown("---")
        st.subheader("步骤 4：可视化验证 ROI 位置")

        visualizer = ROIVisualizer(image_for_viz)
        rois = st.session_state.adv_rois
        handler = st.session_state.adv_roi_handler
        roi_names = handler.get_roi_names(rois)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**选择要显示的 ROI**")
            selected_rois = st.multiselect(
                "ROI 列表",
                roi_names,
                default=roi_names,
                key='adv_selected_rois'
            )

            num_slices = visualizer.image_array.shape[0]
            slice_index = st.slider(
                "切片索引",
                min_value=0,
                max_value=num_slices - 1,
                value=num_slices // 2,
                key='adv_slice_slider'
            )

        with col2:
            fig = visualizer.create_viewer_with_rois(
                rois=rois,
                selected_rois=selected_rois,
                slice_index=slice_index
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**确认无误后继续**")
        if st.button("✅ 确认无误，继续提取特征"):
            st.session_state.adv_verification_complete = True
            st.success("验证完成！可以继续特征提取")

    # Feature extraction
    if st.session_state.adv_verification_complete:
        st.subheader("步骤 5：提取特征")

        if st.session_state.adv_feature_extractor is None:
            st.session_state.adv_feature_extractor = RadiomicsFeatureExtractor()

        rois = st.session_state.adv_rois
        handler = st.session_state.adv_roi_handler
        roi_names = handler.get_roi_names(rois)
        selected_rois = st.multiselect(
            "选择要提特征的 ROI",
            roi_names,
            default=roi_names[:1],
            key='adv_feature_rois'
        )

        if st.button("🚀 提取特征"):
            with st.spinner("正在提取特征..."):
                try:
                    feature_extractor = st.session_state.adv_feature_extractor
                    dicom_series = st.session_state.adv_dicom_series
                    image_to_use = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image

                    if dicom_series is not None:
                        sitk_image = feature_extractor.convert_dicom_series_to_sitk(dicom_series)
                    else:
                        sitk_image = image_to_use

                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
                            with st.spinner(f"正在处理 ROI: {roi.name}"):
                                mask = feature_extractor.convert_roi_to_mask(roi, dicom_series, sitk_image)
                                masks_dict[roi.name] = mask

                    if masks_dict:
                        df_features = feature_extractor.extract_features_for_rois(sitk_image, masks_dict)

                        if not df_features.empty:
                            st.success(f"特征提取完成！共提取了 {len(df_features)} 个 ROI 的特征")

                            st.subheader("特征矩阵")
                            st.dataframe(df_features, use_container_width=True)

                            exporter = ResultsExporter()

                            csv = exporter.to_csv(df_features)
                            st.download_button(
                                label="📥 下载特征矩阵 (CSV)",
                                data=csv,
                                file_name="radiomics_features_advanced.csv",
                                mime="text/csv"
                            )

                            metadata = {
                                '影像名称': getattr(dicom_series, 'series_description', ''),
                                '患者ID': getattr(dicom_series.instances[0].dataset, 'PatientID', '') if dicom_series.instances else '',
                                'ROI数量': len(masks_dict),
                                '特征总数': len(df_features.columns) - 1,
                                '预处理': '是' if st.session_state.adv_preprocessed_image is not None else '否'
                            }
                            excel_bytes = exporter.to_excel(df_features, metadata=metadata)
                            st.download_button(
                                label="📥 下载特征矩阵 (Excel)",
                                data=excel_bytes,
                                file_name="radiomics_features_advanced.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            st.subheader("特征统计摘要")
                            summary_stats = exporter.get_summary_stats(df_features)
                            st.dataframe(summary_stats, use_container_width=True)

                        else:
                            st.warning("未能提取到任何特征")

                except Exception as e:
                    st.error(f"特征提取失败: {e}")
                    st.exception(e)


if __name__ == "__main__":
    main()
