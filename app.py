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
from src.filters import RadiomicsFilterConfig
from src.report_generator import ReportGenerator


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

    upload_method = st.radio("选择上传方式", ["选择文件夹路径（本地部署）", "上传多个文件"], horizontal=True)

    dicom_loaded = False

    if upload_method == "选择文件夹路径（本地部署）":
        dicom_folder = st.text_input("输入 DICOM 文件夹路径", placeholder="/path/to/dicom/folder")

        if dicom_folder and Path(dicom_folder).exists():
            folder_path = Path(dicom_folder)
            dicom_files = list(folder_path.glob("*.dcm")) + list(folder_path.glob("*.DCM")) + \
                         list(folder_path.glob("*.dicom")) + [f for f in folder_path.iterdir() if f.is_file()]

            if dicom_files:
                st.success(f"找到 {len(dicom_files)} 个文件")

                try:
                    # 使用 SimpleITK 直接读取 DICOM 序列
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(str(folder_path))

                    if dicom_names:
                        reader.SetFileNames(dicom_names)
                        sitk_image = reader.Execute()

                        st.session_state.dicom_image = sitk_image
                        dicom_loaded = True
                        st.success(f"DICOM 序列加载成功：{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]} 体素")
                    else:
                        st.error("未找到有效的 DICOM 序列")
                except Exception as e:
                    st.error(f"加载 DICOM 失败: {e}")

    else:
        dicom_upload = st.file_uploader(
            "上传 DICOM 文件（可选中多个）",
            type=['dcm', 'dicom'],
            accept_multiple_files=True,
            key='dicom_uploader'
        )

        if dicom_upload:
            st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for i, uploaded_file in enumerate(dicom_upload):
                    dicom_file = temp_path / f"dicom_{i}.dcm"
                    dicom_file.write_bytes(uploaded_file.getvalue())

                try:
                    # 使用 SimpleITK 读取 DICOM 序列
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(str(temp_path))

                    if dicom_names:
                        reader.SetFileNames(dicom_names)
                        sitk_image = reader.Execute()

                        st.session_state.dicom_image = sitk_image
                        dicom_loaded = True
                        st.success(f"DICOM 序列加载成功：{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]} 体素")
                    else:
                        st.error("未找到有效的 DICOM 序列")
                except Exception as e:
                    st.error(f"加载 DICOM 失败: {e}")

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
                    dicom_image = st.session_state.dicom_image

                    # sitk_image is already a proper 3D volume from SimpleITK
                    sitk_image = dicom_image

                    # Process selected ROIs
                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
                            with st.spinner(f"正在处理 ROI: {roi.name}"):
                                mask = feature_extractor.convert_roi_to_mask(roi, None, sitk_image)
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
                                '影像尺寸': f"{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]}",
                                '体素间距': f"{sitk_image.GetSpacing()[0]:.2f}x{sitk_image.GetSpacing()[1]:.2f}x{sitk_image.GetSpacing()[2]:.2f} mm",
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
    if 'adv_selected_filters' not in st.session_state:
        st.session_state.adv_selected_filters = ['original']
    if 'adv_feature_types' not in st.session_state:
        st.session_state.adv_feature_types = {
            'shape': True,
            'firstorder': True,
            'glcm': True,
            'gldm': True,
            'glrlm': True,
            'glszm': True,
            'ngtdm': True
        }

    st.subheader("步骤 1：上传 DICOM 影像")

    upload_method = st.radio("选择上传方式", ["选择文件夹路径（本地部署）", "上传多个文件"], horizontal=True, key='adv_upload_method')

    dicom_loaded = False

    if upload_method == "选择文件夹路径（本地部署）":
        dicom_folder = st.text_input("输入 DICOM 文件夹路径", placeholder="/path/to/dicom/folder", key='adv_dicom_folder')

        if dicom_folder and Path(dicom_folder).exists():
            folder_path = Path(dicom_folder)

            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(folder_path))

                if dicom_names:
                    reader.SetFileNames(dicom_names)
                    sitk_image = reader.Execute()

                    st.session_state.adv_dicom_image = sitk_image
                    dicom_loaded = True
                    st.success(f"DICOM 序列加载成功：{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]} 体素")
                else:
                    st.error("未找到有效的 DICOM 序列")
            except Exception as e:
                st.error(f"加载 DICOM 失败: {e}")

    else:
        dicom_upload = st.file_uploader(
            "上传 DICOM 文件（可选中多个）",
            type=['dcm', 'dicom'],
            accept_multiple_files=True,
            key='adv_dicom_uploader'
        )

        if dicom_upload:
            st.success(f"已上传 {len(dicom_upload)} 个 DICOM 文件")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for i, uploaded_file in enumerate(dicom_upload):
                    dicom_file = temp_path / f"dicom_{i}.dcm"
                    dicom_file.write_bytes(uploaded_file.getvalue())

                try:
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(str(temp_path))

                    if dicom_names:
                        reader.SetFileNames(dicom_names)
                        sitk_image = reader.Execute()

                        st.session_state.adv_dicom_image = sitk_image
                        dicom_loaded = True
                        st.success(f"DICOM 序列加载成功：{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]} 体素")
                    else:
                        st.error("未找到有效的 DICOM 序列")
                except Exception as e:
                    st.error(f"加载 DICOM 失败: {e}")

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

    # Filter selection
    st.markdown("---")
    st.subheader("步骤 4：滤波器选择")

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        st.markdown("**可用滤波器**")
        filter_options = {k: v['name'] for k, v in RadiomicsFilterConfig.AVAILABLE_FILTERS.items()}
        filter_labels = list(filter_options.values())
        filter_keys = list(filter_options.keys())

        selected_filter_labels = st.multiselect(
            "选择滤波器",
            filter_labels,
            default=[filter_options['original']]
        )

        selected_filters = []
        for label in selected_filter_labels:
            for key, val in filter_options.items():
                if val == label:
                    selected_filters.append(key)
                    break

    with col_filter2:
        st.markdown("**滤波器参数**")
        if 'log' in selected_filters:
            log_sigmas_input = st.text_input("LoG Sigma 值 (逗号分隔)", value="1.0,2.0,3.0")
            log_sigmas = [float(s.strip()) for s in log_sigmas_input.split(',') if s.strip()]
        else:
            log_sigmas = None

        if 'wavelet' in selected_filters:
            wavelet_types = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            selected_wavelet_labels = st.multiselect(
                "小波类型",
                wavelet_types,
                default=['LLL', 'LLH', 'LHL', 'LHH']
            )
        else:
            selected_wavelet_labels = None

    # Build filter settings
    filter_settings = RadiomicsFilterConfig.build_settings(
        enabled_filters=selected_filters,
        log_sigmas=log_sigmas,
        wavelet_types=selected_wavelet_labels
    )

    st.caption(f"已配置 {len(filter_settings.get('enabledImageTypes', []))} 种影像类型")

    # Feature type selection
    st.markdown("---")
    st.subheader("步骤 5：特征类型选择")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        shape_enabled = st.checkbox("形状特征 (Shape)", value=True)
        firstorder_enabled = st.checkbox("一阶特征 (First Order)", value=True)
        glcm_enabled = st.checkbox("GLCM 特征", value=True)

    with col_feat2:
        gldm_enabled = st.checkbox("GLDM 特征", value=True)
        glrlm_enabled = st.checkbox("GLRLM 特征", value=True)
        glszm_enabled = st.checkbox("GLSZM 特征", value=True)

    with col_feat3:
        ngtdm_enabled = st.checkbox("NGTDM 特征", value=True)

    # Build feature classes dict
    feature_classes = {
        'shape': shape_enabled,
        'firstorder': firstorder_enabled,
        'glcm': glcm_enabled,
        'gldm': gldm_enabled,
        'glrlm': glrlm_enabled,
        'glszm': glszm_enabled,
        'ngtdm': ngtdm_enabled
    }

    enabled_feature_count = sum(1 for v in feature_classes.values() if v)
    st.caption(f"已启用 {enabled_feature_count} 个特征类别")

    # Visualization verification step
    image_for_viz = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image
    if dicom_loaded and roi_loaded and image_for_viz is not None and st.session_state.adv_rois is not None:
        st.markdown("---")
        st.subheader("步骤 6：可视化验证 ROI 位置")

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
        st.subheader("步骤 7：提取特征")

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
                    sitk_image = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image

                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
                            with st.spinner(f"正在处理 ROI: {roi.name}"):
                                mask = feature_extractor.convert_roi_to_mask(roi, None, sitk_image)
                                masks_dict[roi.name] = mask

                    if masks_dict:
                        df_features = feature_extractor.extract_features_for_rois(
                            sitk_image,
                            masks_dict,
                            feature_classes=feature_classes,
                            filter_settings=filter_settings
                        )

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
                                '影像尺寸': f"{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]}",
                                '体素间距': f"{sitk_image.GetSpacing()[0]:.2f}x{sitk_image.GetSpacing()[1]:.2f}x{sitk_image.GetSpacing()[2]:.2f} mm",
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

                            # Visualization Report Tab
                            st.markdown("---")
                            st.subheader("可视化报告")

                            report_gen = ReportGenerator()

                            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                                "相关性热力图",
                                "特征分布",
                                "箱线图",
                                "统计摘要"
                            ])

                            with viz_tab1:
                                st.markdown("**特征相关性热力图**")
                                try:
                                    corr_fig = report_gen.create_correlation_heatmap(df_features)
                                    st.plotly_chart(corr_fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"无法生成相关性热力图: {e}")

                            with viz_tab2:
                                st.markdown("**特征分布直方图**")
                                numeric_cols = df_features.select_dtypes(include='number').columns.tolist()
                                if numeric_cols:
                                    selected_feature = st.selectbox(
                                        "选择要查看的特征",
                                        numeric_cols,
                                        key='feature_dist_select'
                                    )
                                    try:
                                        dist_fig = report_gen.create_feature_distribution(df_features, selected_feature)
                                        st.plotly_chart(dist_fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"无法生成分布图: {e}")
                                else:
                                    st.info("没有数值型特征可供显示")

                            with viz_tab3:
                                st.markdown("**ROI 对比箱线图**")
                                numeric_cols_box = df_features.select_dtypes(include='number').columns.tolist()
                                if numeric_cols_box:
                                    selected_feature_box = st.selectbox(
                                        "选择要比较的特征",
                                        numeric_cols_box,
                                        key='box_plot_select'
                                    )
                                    try:
                                        box_fig = report_gen.create_box_plot(df_features, selected_feature_box)
                                        st.plotly_chart(box_fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"无法生成箱线图: {e}")
                                else:
                                    st.info("没有数值型特征可供显示")

                            with viz_tab4:
                                st.markdown("**详细统计摘要**")
                                try:
                                    detailed_summary = report_gen.create_summary_table(df_features)
                                    st.dataframe(detailed_summary, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"无法生成统计摘要: {e}")

                        else:
                            st.warning("未能提取到任何特征")

                except Exception as e:
                    st.error(f"特征提取失败: {e}")
                    st.exception(e)


if __name__ == "__main__":
    main()
