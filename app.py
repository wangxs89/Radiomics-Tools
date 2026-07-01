"""影像组学 Web App 主应用"""
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
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
from src.ui_theme import APP_CSS, get_step_indicator_html


st.set_page_config(
    page_title="Radiomics Tool",
    page_icon="",
    layout="wide"
)

# Inject custom CSS
st.markdown(APP_CSS, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>Radiomics Tool</h1>
        <p>Medical Image Feature Extraction & Analysis Platform</p>
    </div>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown("### Mode")
    mode = st.sidebar.radio("", ["Beginner", "Advanced"],
                          label_visibility="collapsed")

    if mode == "Beginner":
        beginner_mode()
    else:
        advanced_mode()


def load_dicom_and_rtstruct(folder_path: Path, key_prefix: str = ''):
    """Shared DICOM scan + series selection + HU conversion + RTSTRUCT loading.

    Returns:
        (dicom_image, rois, handler, dicom_loaded, roi_loaded) or None
    """
    import pydicom

    all_files = [f for f in folder_path.iterdir() if f.is_file()]
    st.info(f"Found {len(all_files)} files in folder")

    # File-level modality scan
    file_modalities = {}
    for f in all_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            modality = str(getattr(ds, 'Modality', ''))
            if modality:
                file_modalities[f] = modality
        except Exception:
            continue

    # SimpleITK series grouping
    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder_path))

    series_info = []
    if series_ids:
        for sid in series_ids:
            file_names = reader.GetGDCMSeriesFileNames(str(folder_path), sid)
            first_ds = pydicom.dcmread(str(file_names[0]), stop_before_pixels=True)
            modality = str(getattr(first_ds, 'Modality', 'UNKNOWN'))
            series_desc = str(getattr(first_ds, 'SeriesDescription', ''))
            series_info.append({
                'id': sid, 'files': file_names,
                'modality': modality, 'description': series_desc,
                'count': len(file_names)
            })

    rtstruct_files = [f for f, m in file_modalities.items() if m == 'RTSTRUCT']
    dose_files = [f for f, m in file_modalities.items() if m == 'RTDOSE']

    imaging_series = [s for s in series_info if s['modality'] in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR')]
    dose_series = [s for s in series_info if s['modality'] == 'RTDOSE']
    other_series = [s for s in series_info if s['modality'] not in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR', 'RTDOSE', 'RTSTRUCT')]

    sitk_image = None
    dicom_loaded = False

    if imaging_series:
        st.markdown("**Imaging Series (select):**")
        for s in imaging_series:
            label = f"{s['modality']} - {s['description'] or 'Unnamed'} ({s['count']} slices)"
            st.write(f"- `{label}`")

        if len(imaging_series) > 1:
            selected_idx = st.selectbox(
                "Select imaging series to load",
                range(len(imaging_series)),
                format_func=lambda i: f"{imaging_series[i]['modality']} - {imaging_series[i]['description'] or 'Unnamed'} ({imaging_series[i]['count']} slices)",
                key=f'{key_prefix}series_selector'
            )
        else:
            selected_idx = 0

        selected = imaging_series[selected_idx]
        reader.SetFileNames(selected['files'])
        sitk_image = reader.Execute()

        first_ds = pydicom.dcmread(str(selected['files'][0]), stop_before_pixels=True)
        slope = float(getattr(first_ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(first_ds, 'RescaleIntercept', 0.0))

        original_spacing = sitk_image.GetSpacing()
        original_origin = sitk_image.GetOrigin()
        original_dim = sitk_image.GetDimension()

        arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * slope + intercept

        new_image = sitk.GetImageFromArray(arr)
        new_image.SetSpacing(original_spacing)
        new_image.SetOrigin(original_origin)
        if new_image.GetDimension() == original_dim:
            new_image.SetDirection(sitk_image.GetDirection())
        sitk_image = new_image

        dicom_loaded = True
        st.success(f"Loaded: {selected['modality']} - {selected['description'] or 'Unnamed'} ({sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]})")
        st.write(f"**HU range:** {arr.min():.0f} ~ {arr.max():.0f}")
    else:
        st.warning("No imaging series found (CT/MR/PET etc.)")

    if dose_series:
        st.markdown("**Dose Files (RTDOSE):**")
        for s in dose_series:
            st.write(f"- `{s['description'] or 'RTDOSE'}` ({s['count']} slices)")
        st.session_state[f'{key_prefix}dose_series'] = dose_series
        st.info("Dose files identified — available for dose analysis")

    if other_series:
        with st.expander(f"Other series ({len(other_series)}, skipped)"):
            for s in other_series:
                st.write(f"- {s['modality']}: {s['description'] or 'Unnamed'} ({s['count']} slices)")

    # RTSTRUCT
    rois = None
    handler = None
    roi_loaded = False

    if rtstruct_files:
        st.markdown("**RTSTRUCT Files Found:**")
        for rf in rtstruct_files:
            try:
                ds = pydicom.dcmread(str(rf), stop_before_pixels=True)
                desc = str(getattr(ds, 'StructureSetLabel', getattr(ds, 'SeriesDescription', rf.name)))
                st.write(f"- `{desc}`")
            except Exception:
                st.write(f"- `{rf.name}`")

        handler = ROIHandler()
        rois = handler.load_rtstruct(str(rtstruct_files[0]))

        if rois:
            roi_loaded = True
            roi_names = handler.get_roi_names(rois)
            st.success(f"RTSTRUCT loaded: {len(rois)} ROIs ({', '.join(roi_names)})")
        else:
            st.warning("No ROI structures found in RTSTRUCT file")
    else:
        st.warning("No RTSTRUCT file found")

    # NIfTI fallback
    nii_files = list(folder_path.glob("*.nii")) + list(folder_path.glob("*.nii.gz"))
    if nii_files and not roi_loaded:
        st.markdown("**NIfTI Mask Files Found:**")
        for nf in nii_files:
            st.write(f"- `{nf.name}`")
        st.info("NIfTI mask support coming soon")

    return sitk_image, rois, handler, dicom_loaded, roi_loaded


def beginner_mode():
    st.markdown("### Beginner Mode")
    st.caption("Select folder, verify ROI, extract features — three simple steps.")

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
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None

    # Determine current step for indicator
    if st.session_state.verification_complete:
        current_step = 2
    elif st.session_state.dicom_image is not None and st.session_state.rois is not None:
        current_step = 1
    else:
        current_step = 0

    st.markdown(get_step_indicator_html(["Select Data", "Verify ROI", "Extract"], current_step),
                unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Select Data Folder")
    st.caption("Select a folder containing all DICOM files (images + RTSTRUCT)")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Browse Folder", key='pick_folder_btn'):
            import subprocess
            result = subprocess.run([
                'osascript', '-e',
                'set folderPath to POSIX path of (choose folder with prompt "Select DICOM Data Folder")'
            ], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                st.session_state.dicom_folder = result.stdout.strip()
                st.rerun()

    with col1:
        dicom_folder = st.text_input(
            "Folder path (or type manually)",
            value=st.session_state.get('dicom_folder', ''),
            key='dicom_folder_input'
        )
        if dicom_folder:
            st.session_state.dicom_folder = dicom_folder

    dicom_loaded = False
    roi_loaded = False

    if dicom_folder and Path(dicom_folder).exists():
        folder_path = Path(dicom_folder)
        sitk_image, rois, handler, dicom_loaded, roi_loaded = load_dicom_and_rtstruct(folder_path, key_prefix='')

        if sitk_image is not None:
            st.session_state.dicom_image = sitk_image
        if rois is not None:
            st.session_state.rois = rois
            st.session_state.roi_handler = handler
    elif dicom_folder:
        st.error(f"Folder does not exist: {dicom_folder}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Visualization verification step
    if dicom_loaded and roi_loaded and st.session_state.dicom_image is not None and st.session_state.rois is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Verify ROI Position")

        visualizer = ROIVisualizer(st.session_state.dicom_image)
        rois = st.session_state.rois
        roi_names = st.session_state.roi_handler.get_roi_names(rois)

        # ROI selection
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Select ROI**")
            selected_rois = st.multiselect(
                "ROI list",
                roi_names,
                default=roi_names
            )

            # 4D volume selector
            volume_index = 0
            if visualizer.is_4d and visualizer.num_volumes > 1:
                volume_index = st.slider(
                    "Volume index",
                    min_value=0,
                    max_value=visualizer.num_volumes - 1,
                    value=0,
                    key='vol_slider'
                )

            # Slice selector
            num_slices = visualizer.num_slices
            if num_slices > 1:
                slice_index = st.slider(
                    "Slice index",
                    min_value=0,
                    max_value=num_slices - 1,
                    value=num_slices // 2
                )
            else:
                slice_index = 0

            # Window/Level controls
            st.markdown("**Window / Level**")
            preset = st.selectbox("Preset", ["Soft Tissue", "Lung", "Bone", "Brain", "Custom"], key='wl_preset')
            presets = {
                "Soft Tissue": (40, 400),
                "Lung": (-600, 1500),
                "Bone": (400, 1800),
                "Brain": (40, 80),
            }
            if preset != "Custom":
                wc, ww = presets[preset]
            else:
                wc = st.number_input("Level (center)", value=40, step=10, key='wc_val')
                ww = st.number_input("Width", value=400, step=50, key='ww_val')

        # Display visualization
        with col2:
            fig = visualizer.create_viewer_with_rois(
                rois=rois,
                selected_rois=selected_rois,
                slice_index=slice_index,
                volume_index=volume_index,
                window_center=wc,
                window_width=ww
            )
            st.pyplot(fig)

            # 显示切片数据调试信息
            stats = visualizer.get_slice_stats(volume_index, slice_index)
            st.caption(f"Data: min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}, shape={stats['shape']}")

        # Confirmation button
        st.markdown("#### Confirm & Continue")
        if st.button("Confirm ROI, Extract Features"):
            st.session_state.verification_complete = True
            st.success("Verification complete! Proceeding to feature extraction.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Show feature extraction if verification is complete
    if st.session_state.verification_complete:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Extract Features")

        # Initialize feature extractor
        if st.session_state.feature_extractor is None:
            st.session_state.feature_extractor = RadiomicsFeatureExtractor()

        rois = st.session_state.rois
        roi_names = st.session_state.roi_handler.get_roi_names(rois)
        selected_rois = st.multiselect(
            "Select ROIs for feature extraction",
            roi_names,
            default=roi_names[:1],
            key='feature_rois'
        )

        if st.button("Extract Features"):
            with st.spinner("Extracting features..."):
                try:
                    feature_extractor = st.session_state.feature_extractor
                    sitk_image = st.session_state.dicom_image

                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
                            mask = feature_extractor.convert_roi_to_mask(roi, None, sitk_image)
                            masks_dict[roi.name] = mask

                    if masks_dict:
                        df_features = feature_extractor.extract_features_for_rois(sitk_image, masks_dict)
                        if not df_features.empty:
                            st.session_state.features_df = df_features
                            st.session_state.features_metadata = {
                                'Image size': f"{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]}",
                                'Voxel spacing': f"{sitk_image.GetSpacing()[0]:.2f}x{sitk_image.GetSpacing()[1]:.2f}x{sitk_image.GetSpacing()[2]:.2f} mm",
                                'ROI count': len(masks_dict),
                                'Total features': len(df_features.columns) - 1,
                            }
                            st.success(f"Feature extraction complete — {len(df_features)} ROI feature rows")
                            st.rerun()
                        else:
                            st.warning("No features could be extracted")

                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.exception(e)

    if st.session_state.features_df is not None:
        df_features = st.session_state.features_df
        metadata = st.session_state.get('features_metadata', {})

        st.subheader("Feature Matrix")
        st.dataframe(df_features, use_container_width=True)

        exporter = ResultsExporter()

        csv = exporter.to_csv(df_features)
        st.download_button(
            label="Download Feature Matrix (CSV)",
            data=csv,
            file_name="radiomics_features.csv",
            mime="text/csv"
        )

        excel_bytes = exporter.to_excel(df_features, metadata=metadata)
        st.download_button(
            label="Download Feature Matrix (Excel)",
            data=excel_bytes,
            file_name="radiomics_features.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("Feature Summary Statistics")
        summary_stats = exporter.get_summary_stats(df_features)
        st.dataframe(summary_stats, use_container_width=True)


def advanced_mode():
    st.markdown("### Advanced Mode")
    st.caption("Full control: preprocessing, filters, custom features.")


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
    if 'adv_features_df' not in st.session_state:
        st.session_state.adv_features_df = None

    # Determine current step
    if st.session_state.adv_verification_complete:
        adv_step = 5
    elif st.session_state.adv_dicom_image is not None and st.session_state.adv_rois is not None:
        adv_step = 4
    elif st.session_state.adv_dicom_image is not None:
        adv_step = 1
    else:
        adv_step = 0

    st.markdown(get_step_indicator_html(["Data", "Preprocess", "Filters", "Features", "Verify", "Extract"], adv_step),
                unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Select Data Folder")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Browse Folder", key='adv_pick_folder_btn'):
            import subprocess
            result = subprocess.run([
                'osascript', '-e',
                'set folderPath to POSIX path of (choose folder with prompt "Select DICOM Data Folder")'
            ], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                st.session_state.adv_dicom_folder = result.stdout.strip()
                st.rerun()

    with col1:
        dicom_folder = st.text_input(
            "Folder path (or type manually)",
            value=st.session_state.get('adv_dicom_folder', ''),
            key='adv_dicom_folder_input'
        )
        if dicom_folder:
            st.session_state.adv_dicom_folder = dicom_folder

    dicom_loaded = False
    roi_loaded = False

    if dicom_folder and Path(dicom_folder).exists():
        folder_path = Path(dicom_folder)
        sitk_image, rois, handler, dicom_loaded, roi_loaded = load_dicom_and_rtstruct(folder_path, key_prefix='adv_')

        if sitk_image is not None:
            st.session_state.adv_dicom_image = sitk_image
        if rois is not None:
            st.session_state.adv_rois = rois
            st.session_state.adv_roi_handler = handler

    # Preprocessing options
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 1. Image Preprocessing")

    preprocess_enabled = st.checkbox("Enable preprocessing", value=False)

    preprocessor = ImagePreprocessor()
    original_image = st.session_state.adv_dicom_image
    processed_image = original_image

    if preprocess_enabled and original_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Resampling**")
            enable_resample = st.checkbox("Enable resampling", value=False)
            if enable_resample:
                new_spacing_x = st.number_input("X spacing (mm)", value=1.0, step=0.1)
                new_spacing_y = st.number_input("Y spacing (mm)", value=1.0, step=0.1)
                new_spacing_z = st.number_input("Z spacing (mm)", value=1.0, step=0.1)

                if st.button("Apply Resampling"):
                    processed_image = preprocessor.resample_image(
                        processed_image,
                        [new_spacing_x, new_spacing_y, new_spacing_z]
                    )
                    st.success("Resampling complete")

        with col2:
            st.markdown("**Normalization**")
            normalize_method = st.selectbox(
                "Method",
                ["None", "z-score", "min-max", "percentile"]
            )
            if normalize_method != "None":
                if st.button("Apply Normalization"):
                    processed_image = preprocessor.normalize_image(processed_image, method=normalize_method)
                    st.success(f"{normalize_method} normalization complete")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Discretization**")
            enable_discretize = st.checkbox("Enable discretization", value=False)
            if enable_discretize:
                bin_width = st.number_input("Bin width", value=25.0, step=1.0)
                if st.button("Apply Discretization"):
                    processed_image = preprocessor.discretize_image(processed_image, bin_width=bin_width)
                    st.success("Discretization complete")

        if processed_image is not original_image:
            st.session_state.adv_preprocessed_image = processed_image
            st.info("Preprocessed image saved for feature extraction")

    # Filter selection
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 2. Filter Selection")

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        st.markdown("**Available Filters**")
        filter_options = {k: v['name'] for k, v in RadiomicsFilterConfig.AVAILABLE_FILTERS.items()}
        filter_labels = list(filter_options.values())
        filter_keys = list(filter_options.keys())

        selected_filter_labels = st.multiselect(
            "Select filters",
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
        st.markdown("**Filter Parameters**")
        if 'log' in selected_filters:
            log_sigmas_input = st.text_input("LoG Sigma values (comma-separated)", value="1.0,2.0,3.0")
            log_sigmas = [float(s.strip()) for s in log_sigmas_input.split(',') if s.strip()]
        else:
            log_sigmas = None

        if 'wavelet' in selected_filters:
            wavelet_types = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
            selected_wavelet_labels = st.multiselect(
                "Wavelet types",
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

    st.caption(f"{len(filter_settings.get('enabledImageTypes', []))} image types configured")

    # Feature type selection
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 3. Feature Types")

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        shape_enabled = st.checkbox("Shape", value=True)
        firstorder_enabled = st.checkbox("First Order", value=True)
        glcm_enabled = st.checkbox("GLCM", value=True)

    with col_feat2:
        gldm_enabled = st.checkbox("GLDM", value=True)
        glrlm_enabled = st.checkbox("GLRLM", value=True)
        glszm_enabled = st.checkbox("GLSZM", value=True)

    with col_feat3:
        ngtdm_enabled = st.checkbox("NGTDM", value=True)

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
    st.caption(f"{enabled_feature_count} feature classes enabled")

    # Visualization verification step
    image_for_viz = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image
    if dicom_loaded and roi_loaded and image_for_viz is not None and st.session_state.adv_rois is not None:
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 4. Verify ROI Placement")

        visualizer = ROIVisualizer(image_for_viz)
        rois = st.session_state.adv_rois
        handler = st.session_state.adv_roi_handler
        roi_names = handler.get_roi_names(rois)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Select ROIs to display**")
            selected_rois = st.multiselect(
                "ROI list",
                roi_names,
                default=roi_names,
                key='adv_selected_rois'
            )

            num_slices = visualizer.num_slices
            slice_index = st.slider(
                "Slice index",
                min_value=0,
                max_value=num_slices - 1,
                value=num_slices // 2,
                key='adv_slice_slider'
            )

            st.markdown("**Window / Level**")
            adv_preset = st.selectbox("Preset", ["Soft Tissue", "Lung", "Bone", "Brain", "Custom"], key='adv_wl_preset')
            adv_presets = {"Soft Tissue": (40, 400), "Lung": (-600, 1500), "Bone": (400, 1800), "Brain": (40, 80)}
            if adv_preset != "Custom":
                adv_wc, adv_ww = adv_presets[adv_preset]
            else:
                adv_wc = st.number_input("Level", value=40, step=10, key='adv_wc_val')
                adv_ww = st.number_input("Width", value=400, step=50, key='adv_ww_val')

        with col2:
            fig = visualizer.create_viewer_with_rois(
                rois=rois,
                selected_rois=selected_rois,
                slice_index=slice_index,
                window_center=adv_wc,
                window_width=adv_ww
            )
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Once verified, proceed to extraction**")
        if st.button("Confirm and Extract Features"):
            st.session_state.adv_verification_complete = True
            st.success("Verification complete — ready for feature extraction")

    # Feature extraction
    if st.session_state.adv_verification_complete:
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 5. Extract Features")

        if st.session_state.adv_feature_extractor is None:
            st.session_state.adv_feature_extractor = RadiomicsFeatureExtractor()

        rois = st.session_state.adv_rois
        handler = st.session_state.adv_roi_handler
        roi_names = handler.get_roi_names(rois)
        selected_rois = st.multiselect(
            "Select ROIs for feature extraction",
            roi_names,
            default=roi_names[:1],
            key='adv_feature_rois'
        )

        if st.button("Extract Features"):
            with st.spinner("Extracting features..."):
                try:
                    feature_extractor = st.session_state.adv_feature_extractor
                    sitk_image = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image

                    masks_dict = {}
                    for roi in rois:
                        if roi.name in selected_rois:
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
                            st.session_state.adv_features_df = df_features
                            st.session_state.adv_features_metadata = {
                                'Image size': f"{sitk_image.GetSize()[0]}x{sitk_image.GetSize()[1]}x{sitk_image.GetSize()[2]}",
                                'Voxel spacing': f"{sitk_image.GetSpacing()[0]:.2f}x{sitk_image.GetSpacing()[1]:.2f}x{sitk_image.GetSpacing()[2]:.2f} mm",
                                'ROI count': len(masks_dict),
                                'Total features': len(df_features.columns) - 1,
                                'Preprocessing': 'Yes' if st.session_state.adv_preprocessed_image is not None else 'No'
                            }
                            st.success(f"Feature extraction complete — {len(df_features)} ROI feature rows")
                            st.rerun()
                        else:
                            st.warning("No features could be extracted")

                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.exception(e)

    if st.session_state.adv_features_df is not None:
        df_features = st.session_state.adv_features_df
        metadata = st.session_state.get('adv_features_metadata', {})

        st.subheader("Feature Matrix")
        st.dataframe(df_features, use_container_width=True)

        exporter = ResultsExporter()

        csv = exporter.to_csv(df_features)
        st.download_button(
            label="Download Feature Matrix (CSV)",
            data=csv,
            file_name="radiomics_features_advanced.csv",
            mime="text/csv"
        )

        excel_bytes = exporter.to_excel(df_features, metadata=metadata)
        st.download_button(
            label="Download Feature Matrix (Excel)",
            data=excel_bytes,
            file_name="radiomics_features_advanced.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("Feature Summary Statistics")
        summary_stats = exporter.get_summary_stats(df_features)
        st.dataframe(summary_stats, use_container_width=True)

        # Visualization Report Tabs
        st.markdown("---")
        st.subheader("Visualization Report")

        report_gen = ReportGenerator()

        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Correlation Heatmap",
            "Feature Distribution",
            "Box Plot",
            "Summary Statistics"
        ])

        with viz_tab1:
            st.markdown("**Feature Correlation Heatmap**")
            try:
                corr_fig = report_gen.create_correlation_heatmap(df_features)
                st.plotly_chart(corr_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {e}")

        with viz_tab2:
            st.markdown("**Feature Distribution Histogram**")
            numeric_cols = df_features.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected_feature = st.selectbox(
                    "Select feature",
                    numeric_cols,
                    key='feature_dist_select'
                )
                try:
                    dist_fig = report_gen.create_feature_distribution(df_features, selected_feature)
                    st.plotly_chart(dist_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate distribution plot: {e}")
            else:
                st.info("No numeric features available to display")

        with viz_tab3:
            st.markdown("**ROI Comparison Box Plot**")
            numeric_cols_box = df_features.select_dtypes(include='number').columns.tolist()
            if numeric_cols_box:
                selected_feature_box = st.selectbox(
                    "Select feature",
                    numeric_cols_box,
                    key='box_plot_select'
                )
                try:
                    box_fig = report_gen.create_box_plot(df_features, selected_feature_box)
                    st.plotly_chart(box_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate box plot: {e}")
            else:
                st.info("No numeric features available to display")

        with viz_tab4:
            st.markdown("**Detailed Summary Statistics**")
            try:
                detailed_summary = report_gen.create_summary_table(df_features)
                st.dataframe(detailed_summary, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate summary statistics: {e}")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
