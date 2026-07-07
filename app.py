"""Radiomics Web App — Main Application"""
from contextlib import contextmanager
from typing import Optional
import subprocess
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from src.roi_handler import ROIHandler
from src.feature_extractor import RadiomicsFeatureExtractor
from src.roi_visualizer import ROIVisualizer
from src.results_exporter import ResultsExporter
from src.image_preprocessor import ImagePreprocessor
from src.filters import RadiomicsFilterConfig
from src.report_generator import ReportGenerator
from src.dosomics import (
    load_dose_image, extract_dose_features, resample_mask_to_image,
)
from src.statistics import (
    compute_icc, lasso_feature_selection, correlation_analysis,
    icc_reliability_classification,
)
from src.data_store import (
    DEFAULT_DB_PATH, build_modeling_table, clear_database,
    list_saved_results, load_clinical_followup, load_feature_wide_table,
    read_tabular_upload, save_clinical_followup, save_feature_dataframe,
)
from src.modeling import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, candidate_feature_columns,
    infer_task_type, run_kmeans_clustering, run_supervised_models,
)
from src.ui_theme import (
    APP_CSS, get_footer_html, get_header_html, get_hero_banner_html,
    get_sidebar_logo_html, get_sidebar_section_html, get_step_indicator_html,
)


st.set_page_config(page_title="Radiomics Tool", page_icon="", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────
#  Shared Helpers
# ─────────────────────────────────────────────

def init_state(defaults: dict) -> None:
    """Initialize session_state keys with defaults (only if absent)."""
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_extraction_results(key_prefix: str = '') -> None:
    """Clear cached extraction outputs before starting a new run."""
    st.session_state[f'{key_prefix}features_df'] = None
    st.session_state[f'{key_prefix}features_metadata'] = {}
    st.session_state[f'{key_prefix}dose_features_df'] = None


def ensure_feature_extractor(key: str) -> None:
    """Create a feature extractor when session state has no usable instance."""
    extractor = st.session_state.get(key)
    if extractor is None or not hasattr(extractor, 'convert_roi_to_mask'):
        st.session_state[key] = RadiomicsFeatureExtractor()


def get_default_case_id(folder: str) -> str:
    """Infer a case ID from the selected folder path."""
    if not folder:
        return ""
    return Path(folder).name or Path(folder).parent.name


def make_series_label(series_info: dict) -> str:
    modality = series_info.get('modality') or 'IMG'
    description = series_info.get('description') or 'Unnamed'
    count = series_info.get('count', 0)
    return f"{modality} - {description} ({count} slices)"


def load_image_from_series(series_info: dict) -> sitk.Image:
    """Load one selected imaging series as a float SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_info['files'])
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    if out.GetDimension() == image.GetDimension():
        out.SetDirection(image.GetDirection())
    return out


def annotate_feature_rows(df: pd.DataFrame, case_id: str, series_info: dict,
                          feature_kind: str = 'imaging') -> pd.DataFrame:
    """Add case/series metadata columns to a feature matrix."""
    if df is None or df.empty:
        return df
    annotated = df.copy()
    annotated.insert(0, 'FeatureKind', feature_kind)
    annotated.insert(0, 'Modality', series_info.get('modality', ''))
    annotated.insert(0, 'SeriesUID', series_info.get('id', ''))
    annotated.insert(0, 'Series', make_series_label(series_info))
    annotated.insert(0, 'CaseID', case_id)
    return annotated


@contextmanager
def section_card():
    """Context manager for a bordered section container."""
    try:
        card = st.container(border=True)
    except TypeError:
        card = st.container()
    with card:
        yield


def icon_heading(title: str, icon: str, level: int = 3,
                 subtitle: str = "", accent: str = "primary") -> None:
    """Render a reusable heading without raw HTML tags."""
    symbols = {
        'activity': '▰',
        'bar-chart': '▥',
        'brain-circuit': '◇',
        'check-circle': '✓',
        'clipboard-list': '☷',
        'database': '▦',
        'eye': '◎',
        'file-spreadsheet': '▤',
        'filter': '▽',
        'folder-open': '▣',
        'grid': '▦',
        'layers': '▧',
        'line-chart': '⌁',
        'network': '◇',
        'save': '▣',
        'scan': '□',
        'settings': '◈',
        'sliders': '≡',
        'target': '⊙',
        'upload': '⇧',
    }
    heading_level = "#" * min(max(level, 2), 5)
    st.markdown(f"{heading_level} {symbols.get(icon, '•')} {title}")
    if subtitle:
        st.caption(subtitle)


def open_browser_folder(prompt: str) -> Optional[str]:
    """Open macOS native folder picker. Returns POSIX path or None."""
    result = subprocess.run(
        ['osascript', '-e',
         f'set folderPath to POSIX path of (choose folder with prompt "{prompt}")'],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


# ─────────────────────────────────────────────
#  Shared: Folder picker + DICOM loading
# ─────────────────────────────────────────────

def load_dicom_and_rtstruct(folder_path: Path, key_prefix: str = ''):
    """Shared DICOM scan + series selection + HU conversion + RTSTRUCT loading.

    Returns:
        (sitk_image, rois, handler, dicom_loaded, roi_loaded)
    """
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

            # Sort files by Z position (ImagePositionPatient[2])
            def get_z_pos(fname):
                try:
                    ds = pydicom.dcmread(str(fname), stop_before_pixels=True)
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    return 0.0

            file_names = sorted(file_names, key=get_z_pos)

            series_info.append({
                'id': sid, 'files': file_names,
                'modality': modality, 'description': series_desc,
                'count': len(file_names),
            })

    rtstruct_files = [f for f, m in file_modalities.items() if m == 'RTSTRUCT']
    imaging_series = [s for s in series_info
                      if s['modality'] in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR')]
    for idx, series in enumerate(imaging_series):
        series['index'] = idx
    dose_series = [s for s in series_info if s['modality'] == 'RTDOSE']
    other_series = [s for s in series_info
                    if s['modality'] not in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR', 'RTDOSE', 'RTSTRUCT')]

    sitk_image = None
    dicom_loaded = False

    if imaging_series:
        st.session_state[f'{key_prefix}imaging_series'] = imaging_series
        icon_heading(
            "Imaging Series", "layers", level=5,
            subtitle="Select one series for viewing; multiple series can be extracted later.",
        )
        for s in imaging_series:
            st.write(f"- `{make_series_label(s)}`")

        if len(imaging_series) > 1:
            selected_idx = st.selectbox(
                "Select imaging series to load",
                range(len(imaging_series)),
                format_func=lambda i: (
                    f"{imaging_series[i]['modality']} - "
                    f"{imaging_series[i]['description'] or 'Unnamed'} "
                    f"({imaging_series[i]['count']} slices)"
                ),
                key=f'{key_prefix}series_selector',
            )
        else:
            selected_idx = 0

        selected = imaging_series[selected_idx]
        st.session_state[f'{key_prefix}selected_series_index'] = selected_idx
        st.session_state[f'{key_prefix}selected_series_info'] = selected
        sitk_image = load_image_from_series(selected)
        arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

        dicom_loaded = True
        st.success(
            f"Loaded: {make_series_label(selected)} "
            f"({sitk_image.GetSize()[0]}×{sitk_image.GetSize()[1]}×{sitk_image.GetSize()[2]})"
        )
        st.write(f"**HU range:** {max(arr.min(), -1024):.0f} ~ {arr.max():.0f}")
    else:
        st.warning("No imaging series found (CT/MR/PET etc.)")

    # Dose loading
    dose_image = None
    if dose_series:
        icon_heading("Dose Files (RTDOSE)", "activity", level=5, accent="secondary")
        for s in dose_series:
            st.write(f"- `{s['description'] or 'RTDOSE'}` ({s['count']} slices)")
        st.session_state[f'{key_prefix}dose_series'] = dose_series

        if len(dose_series) == 1:
            selected_dose = dose_series[0]
        else:
            dose_idx = st.selectbox(
                "Select dose series", range(len(dose_series)),
                format_func=lambda i: f"{dose_series[i]['description'] or 'RTDOSE'} ({dose_series[i]['count']} slices)",
                key=f'{key_prefix}dose_selector',
            )
            selected_dose = dose_series[dose_idx]

        try:
            dose_image = load_dose_image(selected_dose['files'])
            dose_arr = sitk.GetArrayFromImage(dose_image)
            max_dose = float(np.max(dose_arr))
            st.success(f"Dose loaded: {dose_image.GetSize()[0]}×{dose_image.GetSize()[1]}×{dose_image.GetSize()[2]}, max={max_dose:.1f} Gy")
            st.caption(
                f"Dose geometry — spacing: {dose_image.GetSpacing()[0]:.2f}×{dose_image.GetSpacing()[1]:.2f}×{dose_image.GetSpacing()[2]:.2f} mm, "
                f"origin: ({dose_image.GetOrigin()[0]:.1f}, {dose_image.GetOrigin()[1]:.1f}, {dose_image.GetOrigin()[2]:.1f})"
            )
        except Exception as e:
            st.warning(f"Could not load dose image: {e}")

    if other_series:
        with st.expander(f"Other series ({len(other_series)}, skipped)"):
            for s in other_series:
                st.write(f"- {s['modality']}: {s['description'] or 'Unnamed'} ({s['count']} slices)")

    # RTSTRUCT
    rois = None
    handler = None
    roi_loaded = False

    if rtstruct_files:
        icon_heading("RTSTRUCT Files Found", "target", level=5, accent="secondary")
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
        icon_heading("NIfTI Mask Files Found", "layers", level=5, accent="secondary")
        for nf in nii_files:
            st.write(f"- `{nf.name}`")
        st.info("NIfTI mask support coming soon")

    return sitk_image, rois, handler, dicom_loaded, roi_loaded, dose_image


def folder_picker(key_prefix: str = '') -> dict:
    """Render folder picker UI and load DICOM data.

    Returns dict with keys: dicom_image, rois, handler, dicom_loaded, roi_loaded.
    Also writes to the appropriate session_state keys.
    """
    ss_key = f'{key_prefix}dicom_folder'

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Browse Folder", key=f'{key_prefix}pick_folder_btn'):
            path = open_browser_folder("Select DICOM Data Folder")
            if path:
                st.session_state[ss_key] = path
                st.rerun()

    with col1:
        dicom_folder = st.text_input(
            "Folder path (or type manually)",
            value=st.session_state.get(ss_key, ''),
            key=f'{key_prefix}dicom_folder_input',
        )
        if dicom_folder:
            st.session_state[ss_key] = dicom_folder

    result = {'dicom_image': None, 'rois': None, 'handler': None,
              'dicom_loaded': False, 'roi_loaded': False, 'dose_image': None}

    if dicom_folder and Path(dicom_folder).exists():
        sitk_image, rois, handler, dicom_loaded, roi_loaded, dose_image = load_dicom_and_rtstruct(
            Path(dicom_folder), key_prefix=key_prefix,
        )
        result = {
            'dicom_image': sitk_image,
            'rois': rois,
            'handler': handler,
            'dicom_loaded': dicom_loaded,
            'roi_loaded': roi_loaded,
            'dose_image': dose_image,
        }
        if sitk_image is not None:
            st.session_state[f'{key_prefix}dicom_image'] = sitk_image
        if rois is not None:
            st.session_state[f'{key_prefix}rois'] = rois
            st.session_state[f'{key_prefix}roi_handler'] = handler
        if dose_image is not None:
            st.session_state[f'{key_prefix}dose_image'] = dose_image
    elif dicom_folder:
        st.error(f"Folder does not exist: {dicom_folder}")

    return result


# ─────────────────────────────────────────────
#  Shared: ROI verification visualization
# ────────────────────────────────────────────

def render_visualization(image, rois, handler, key_prefix: str = '') -> bool:
    """Render ROI verification UI. Returns True if user confirmed."""
    verified_key = f'{key_prefix}verified'
    roi_names = handler.get_roi_names(rois)

    with section_card():
        icon_heading("Verify ROI Placement", "eye", level=4)

        col1, col2 = st.columns([1, 3])
        with col1:
            icon_heading("Select ROIs to Display", "target", level=5, accent="secondary")
            selected_rois = st.multiselect(
                "ROI list", roi_names, default=roi_names,
                key=f'{key_prefix}selected_rois',
            )

            visualizer = ROIVisualizer(image)
            num_slices = visualizer.num_slices
            slice_index = st.slider(
                "Slice index", 0, num_slices - 1, num_slices // 2,
                key=f'{key_prefix}slice_slider',
            )

            icon_heading("Window / Level", "sliders", level=5, accent="secondary")
            preset = st.selectbox(
                "Preset", ["Soft Tissue", "Lung", "Bone", "Brain", "Custom"],
                key=f'{key_prefix}wl_preset',
            )
            presets = {"Soft Tissue": (40, 400), "Lung": (-600, 1500),
                       "Bone": (400, 1800), "Brain": (40, 80)}
            if preset != "Custom":
                wc, ww = presets[preset]
            else:
                wc = st.number_input("Level", value=40, step=10, key=f'{key_prefix}wc_val')
                ww = st.number_input("Width", value=400, step=50, key=f'{key_prefix}ww_val')

        with col2:
            fig = visualizer.create_viewer_with_rois(
                rois=rois, selected_rois=selected_rois,
                slice_index=slice_index, window_center=wc, window_width=ww,
            )
            st.pyplot(fig)
            stats = visualizer.get_slice_stats(0, slice_index)
            display_min = max(stats['min'], -1024)
            st.caption(
                f"Data: min={display_min:.1f}, max={stats['max']:.1f}, "
                f"mean={stats['mean']:.1f}, shape={stats['shape']}"
            )

        st.markdown("---")
        if st.button("Confirm and Proceed", key=f'{key_prefix}confirm_btn'):
            st.session_state[verified_key] = True
            st.success("Verification complete!")

    return st.session_state.get(verified_key, False)


# ────────────────────────────────────────────
#  Shared: Feature extraction
# ─────────────────────────────────────────────

def extract_feature_dataframe(feature_extractor, sitk_image, rois, selected_rois,
                              **extract_kwargs) -> pd.DataFrame:
    """Extract a feature matrix for selected ROIs without writing session state."""
    masks_dict = {}
    for roi in rois:
        if roi.name in selected_rois:
            mask = feature_extractor.convert_roi_to_mask(roi, None, sitk_image)
            masks_dict[roi.name] = mask

    if not masks_dict:
        return pd.DataFrame()

    return feature_extractor.extract_features_for_rois(
        sitk_image, masks_dict, **extract_kwargs,
    )


def run_imaging_batch_extraction(
    feature_extractor,
    rois,
    selected_rois,
    *,
    imaging_series,
    selected_indices,
    current_index,
    current_image,
    case_id,
    key_prefix: str = '',
    **extract_kwargs,
) -> bool:
    """Extract imaging features for one or more imaging series."""
    frames = []
    for idx in selected_indices:
        series_info = imaging_series[idx]
        image = current_image if idx == current_index else load_image_from_series(series_info)
        df = extract_feature_dataframe(
            feature_extractor, image, rois, selected_rois, **extract_kwargs,
        )
        if df is not None and not df.empty:
            frames.append(annotate_feature_rows(df, case_id, series_info, 'imaging'))

    if not frames:
        st.warning("No imaging features could be extracted for the selected series.")
        return False

    df_features = pd.concat(frames, ignore_index=True)
    st.session_state[f'{key_prefix}features_df'] = df_features
    st.session_state[f'{key_prefix}features_metadata'] = {
        'Case ID': case_id,
        'Series count': len(frames),
        'ROI rows': len(df_features),
        'Total columns': len(df_features.columns),
    }
    st.success(f"Feature extraction complete — {len(df_features)} ROI-series rows")
    return True


def run_extraction(feature_extractor, sitk_image, rois, selected_rois,
                   key_prefix: str = '', **extract_kwargs) -> bool:
    """Run feature extraction, save results to session_state. Returns success bool."""
    try:
        df_features = extract_feature_dataframe(
            feature_extractor, sitk_image, rois, selected_rois, **extract_kwargs,
        )
        if df_features.empty:
            st.warning("No ROIs selected for extraction")
            return False

        st.session_state[f'{key_prefix}features_df'] = df_features
        st.session_state[f'{key_prefix}features_metadata'] = {
            'Image size': f"{sitk_image.GetSize()[0]}×{sitk_image.GetSize()[1]}×{sitk_image.GetSize()[2]}",
            'Voxel spacing': (
                f"{sitk_image.GetSpacing()[0]:.2f}×"
                f"{sitk_image.GetSpacing()[1]:.2f}×"
                f"{sitk_image.GetSpacing()[2]:.2f} mm"
            ),
            'ROI count': len(df_features),
            'Total features': len(df_features.columns) - 1,
        }
        if key_prefix == 'adv_':
            st.session_state[f'{key_prefix}features_metadata']['Preprocessing'] = (
                'Yes' if st.session_state.get('adv_preprocessed_image') is not None else 'No'
            )
        st.success(f"Feature extraction complete — {len(df_features)} ROI feature rows")
        return True

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.exception(e)
        return False


# ─────────────────────────────────────────────
#  Shared: Results display
# ─────────────────────────────────────────────

def render_results(df_features, metadata: dict, key_prefix: str = '',
                   show_report: bool = False) -> None:
    """Display feature matrix, downloads, summary stats, and optional report tabs."""
    suffix = '_advanced' if key_prefix == 'adv_' else ''

    with section_card():
        icon_heading("Feature Matrix", "file-spreadsheet", level=4)
        st.dataframe(df_features, use_container_width=True)

        exporter = ResultsExporter()

        csv = exporter.to_csv(df_features)
        st.download_button(
            label="Download Feature Matrix (CSV)", data=csv,
            file_name=f"radiomics_features{suffix}.csv", mime="text/csv",
        )

        excel_bytes = exporter.to_excel(df_features, metadata=metadata)
        st.download_button(
            label="Download Feature Matrix (Excel)", data=excel_bytes,
            file_name=f"radiomics_features{suffix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        icon_heading("Feature Summary Statistics", "bar-chart", level=4, accent="secondary")
        summary_stats = exporter.get_summary_stats(df_features)
        st.dataframe(summary_stats, use_container_width=True)

    if show_report:
        render_visualization_report(df_features, key_prefix)


def render_visualization_report(df_features, key_prefix: str = '') -> None:
    """Render the 4-tab visualization report (advanced mode only)."""
    with section_card():
        st.markdown("---")
        icon_heading("Visualization Report", "line-chart", level=4)

        report_gen = ReportGenerator()

        tab1, tab2, tab3, tab4 = st.tabs([
            "Correlation Heatmap", "Feature Distribution", "Box Plot", "Summary Statistics",
        ])

        with tab1:
            icon_heading("Feature Correlation Heatmap", "grid", level=5, accent="secondary")
            try:
                fig = report_gen.create_correlation_heatmap(df_features)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {e}")

        with tab2:
            icon_heading("Feature Distribution Histogram", "bar-chart", level=5, accent="secondary")
            numeric_cols = df_features.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected = st.selectbox("Select feature", numeric_cols, key=f'{key_prefix}dist_sel')
                try:
                    fig = report_gen.create_feature_distribution(df_features, selected)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate distribution plot: {e}")
            else:
                st.info("No numeric features available to display")

        with tab3:
            icon_heading("ROI Comparison Box Plot", "target", level=5, accent="secondary")
            numeric_cols = df_features.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected = st.selectbox("Select feature", numeric_cols, key=f'{key_prefix}box_sel')
                try:
                    fig = report_gen.create_box_plot(df_features, selected)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate box plot: {e}")
            else:
                st.info("No numeric features available to display")

        with tab4:
            icon_heading("Detailed Summary Statistics", "clipboard-list", level=5, accent="secondary")
            try:
                summary = report_gen.create_summary_table(df_features)
                st.dataframe(summary, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate summary statistics: {e}")


def render_database_save_controls(key_prefix: str = '') -> None:
    """Render buttons for saving current extraction outputs to the temp database."""
    features_df = st.session_state.get(f'{key_prefix}features_df')
    dose_df = st.session_state.get(f'{key_prefix}dose_features_df')
    if (features_df is None or features_df.empty) and (dose_df is None or dose_df.empty):
        return

    with section_card():
        icon_heading("Save to Temporary Database", "save", level=4)
        st.caption(f"Database: `{DEFAULT_DB_PATH}`")

        folder_key = f'{key_prefix}dicom_folder'
        default_case_id = st.session_state.get(f'{key_prefix}case_id') or get_default_case_id(
            st.session_state.get(folder_key, '')
        )
        case_id = st.text_input(
            "Case ID",
            value=default_case_id,
            key=f'{key_prefix}save_case_id',
        ).strip()
        st.session_state[f'{key_prefix}case_id'] = case_id

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Imaging Features", key=f'{key_prefix}save_img_btn', disabled=not case_id):
                rows = save_feature_dataframe(
                    features_df, case_id=case_id, feature_kind='imaging',
                    metadata=st.session_state.get(f'{key_prefix}features_metadata', {}),
                )
                st.success(f"Saved {rows} imaging ROI rows to database.")

        with col2:
            if st.button("Save Dose Features", key=f'{key_prefix}save_dose_btn', disabled=not case_id):
                if dose_df is None or dose_df.empty:
                    st.warning("No dose features available to save.")
                else:
                    dose_series = {
                        'id': 'RTDOSE',
                        'modality': 'RTDOSE',
                        'description': 'Dose',
                        'count': 1,
                    }
                    annotated_dose = annotate_feature_rows(dose_df, case_id, dose_series, 'dose')
                    rows = save_feature_dataframe(
                        annotated_dose, case_id=case_id, feature_kind='dose',
                        metadata={'source': 'RTDOSE'},
                    )
                    st.success(f"Saved {rows} dose ROI rows to database.")


# ─────────────────────────────────────────────
#  Statistical Analysis UI
# ─────────────────────────────────────────────

def render_statistical_analysis(df_features: pd.DataFrame, key_prefix: str = ''):
    """Render statistical analysis section: ICC, LASSO, Correlation."""
    import plotly.express as px
    import plotly.graph_objects as go

    with section_card():
        icon_heading(
            "Statistical Analysis", "bar-chart", level=4,
            subtitle="ICC reliability, LASSO feature selection, and correlation analysis",
            accent="secondary",
        )

        # Check if we have enough data
        if df_features is None or df_features.empty:
            st.info("No feature data available. Please extract features first.")
            return

        # Separate ROI column from numeric features
        roi_col = 'ROI' if 'ROI' in df_features.columns else None
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric features for analysis.")
            return

        # Analysis type selector
        analysis_type = st.radio(
            "Select analysis type",
            ["Correlation Analysis", "LASSO Feature Selection", "ICC Reliability"],
            horizontal=True,
            key=f'{key_prefix}stats_type',
        )

        if analysis_type == "Correlation Analysis":
            icon_heading("Correlation Analysis", "line-chart", level=5, accent="secondary")
            st.caption("Identify highly correlated feature pairs and reduce redundancy")

            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox(
                    "Correlation method",
                    ["pearson", "spearman", "kendall"],
                    key=f'{key_prefix}corr_method',
                )
            with col2:
                threshold = st.slider(
                    "High correlation threshold",
                    min_value=0.5, max_value=1.0, value=0.9, step=0.05,
                    key=f'{key_prefix}corr_threshold',
                )

            if st.button("Run Correlation Analysis", key=f'{key_prefix}corr_btn'):
                with st.spinner("Computing correlations..."):
                    results = correlation_analysis(df_features[numeric_cols], method, threshold)

                corr_matrix = results['correlation_matrix']
                high_corr = results['high_correlations']
                clusters = results['feature_clusters']

                # Display correlation matrix heatmap
                icon_heading("Correlation Matrix Heatmap", "grid", level=5, accent="secondary")
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f"Feature Correlation ({method})",
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Display high correlations
                if len(high_corr) > 0:
                    icon_heading(f"Highly Correlated Pairs (|r| ≥ {threshold})", "network", level=5, accent="secondary")
                    st.dataframe(high_corr, use_container_width=True)
                    st.caption(f"Found {len(high_corr)} highly correlated pairs")
                else:
                    st.success(f"No pairs with |correlation| ≥ {threshold}")

                # Display clusters
                if clusters:
                    icon_heading("Feature Clusters", "network", level=5, subtitle="Highly correlated feature groups")
                    for i, cluster in enumerate(clusters, 1):
                        st.markdown(f"- **Cluster {i}**: {', '.join(cluster[:5])}" +
                                   (f" ... (+{len(cluster)-5} more)" if len(cluster) > 5 else ""))

        elif analysis_type == "LASSO Feature Selection":
            icon_heading("LASSO Feature Selection", "filter", level=5, accent="secondary")
            st.caption("Select most important features using L1 regularization")

            # Need target variable
            st.info("LASSO requires a target variable (outcome). "
                   "Please provide a CSV file with outcomes or use ROI names as proxy.")

            # Option to use ROI as target (for demo)
            use_roi_as_target = st.checkbox(
                "Use ROI name as target (for demonstration)",
                value=False,
                key=f'{key_prefix}lasso_roi_target',
            )

            if use_roi_as_target and roi_col:
                # Encode ROI names as numeric
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(df_features[roi_col]))
                    st.caption(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                except ModuleNotFoundError:
                    y = None
                    st.error("LASSO requires scikit-learn. Install dependencies from requirements.txt.")
            else:
                # Upload target CSV
                target_file = st.file_uploader(
                    "Upload target variable CSV",
                    type=['csv'],
                    key=f'{key_prefix}lasso_target',
                )
                y = None
                if target_file is not None:
                    target_df = pd.read_csv(target_file)
                    if len(target_df) == len(df_features):
                        # Try to find a numeric column
                        numeric_target = target_df.select_dtypes(include=[np.number])
                        if len(numeric_target.columns) > 0:
                            y = numeric_target.iloc[:, 0]
                            st.caption(f"Using '{numeric_target.columns[0]}' as target")
                        else:
                            st.error("Target file must have at least one numeric column")
                    else:
                        st.error(f"Target file has {len(target_df)} rows, expected {len(df_features)}")

            col1, col2 = st.columns(2)
            with col1:
                cv_folds = st.slider("CV folds", 2, 10, 5, key=f'{key_prefix}lasso_cv')
            with col2:
                use_cv = st.checkbox("Auto-select alpha (CV)", value=True, key=f'{key_prefix}lasso_autocv')

            if st.button("Run LASSO", key=f'{key_prefix}lasso_btn'):
                if y is None:
                    st.error("Please provide a target variable")
                else:
                    try:
                        with st.spinner("Running LASSO..."):
                            results = lasso_feature_selection(
                                df_features[numeric_cols], y,
                                alpha=None if use_cv else 0.01,
                                cv_folds=cv_folds,
                            )
                    except ModuleNotFoundError:
                        st.error("LASSO requires scikit-learn. Install dependencies from requirements.txt.")
                        return

                    selected = results['selected_features']
                    coefficients = results['coefficients']

                    st.markdown(f"**Selected {len(selected)} features**")

                    if selected:
                        # Display coefficient plot
                        fig = px.bar(
                            coefficients.head(20),
                            x='Coefficient',
                            y='Feature',
                            orientation='h',
                            title="Top 20 Feature Coefficients",
                            color='Coefficient',
                            color_continuous_scale='RdBu_r',
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        # Display selected features
                        icon_heading("Selected Features", "check-circle", level=5, accent="accent")
                        for feat in selected:
                            coef = coefficients[coefficients['Feature'] == feat]['Coefficient'].values[0]
                            st.markdown(f"- `{feat}` (coef={coef:.4f})")
                    else:
                        st.warning("No features selected. Try adjusting alpha or CV folds.")

                    # Download selected features
                    if selected:
                        selected_df = df_features[['ROI'] + selected] if roi_col else df_features[selected]
                        csv = selected_df.to_csv(index=False)
                        st.download_button(
                            "Download Selected Features (CSV)",
                            data=csv,
                            file_name="lasso_selected_features.csv",
                            mime="text/csv",
                        )

        elif analysis_type == "ICC Reliability":
            icon_heading("ICC Reliability Analysis", "clipboard-list", level=5, accent="secondary")
            st.caption("Assess feature reliability across repeated measurements")

            st.info("ICC requires repeated measurements (e.g., test-retest data). "
                   "Please upload a CSV with paired measurements.")

            # Upload test-retest data
            icc_file = st.file_uploader(
                "Upload test-retest CSV (each subject should have 2 rows)",
                type=['csv'],
                key=f'{key_prefix}icc_upload',
            )

            icc_type = st.selectbox(
                "ICC type",
                ["ICC(2,1)", "ICC(1,1)", "ICC(3,1)", "ICC(2,k)", "ICC(1,k)", "ICC(3,k)"],
                key=f'{key_prefix}icc_type',
            )

            if st.button("Compute ICC", key=f'{key_prefix}icc_btn'):
                if icc_file is None:
                    st.error("Please upload test-retest data")
                else:
                    icc_data = pd.read_csv(icc_file)
                    with st.spinner("Computing ICC..."):
                        icc_results = compute_icc(icc_data, icc_type)

                    # Display ICC results
                    icon_heading("ICC Results", "file-spreadsheet", level=5, accent="secondary")

                    # Add reliability classification
                    icc_results['Reliability'] = icc_results['ICC'].apply(
                        lambda x: icc_reliability_classification(x) if not np.isnan(x) else "N/A"
                    )

                    st.dataframe(icc_results, use_container_width=True)

                    # ICC distribution plot
                    valid_icc = icc_results['ICC'].dropna()
                    if len(valid_icc) > 0:
                        fig = px.histogram(
                            valid_icc,
                            nbins=20,
                            title=f"ICC Distribution ({icc_type})",
                            labels={'value': 'ICC Value', 'count': 'Number of Features'},
                        )
                        fig.add_vline(x=0.75, line_dash="dash", line_color="green",
                                     annotation_text="Good threshold")
                        fig.add_vline(x=0.9, line_dash="dash", line_color="blue",
                                     annotation_text="Excellent threshold")
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary statistics
                        icon_heading("Summary", "bar-chart", level=5, accent="secondary")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean ICC", f"{valid_icc.mean():.3f}")
                        col2.metric("Median ICC", f"{valid_icc.median():.3f}")
                        col3.metric(
                            "Excellent (≥0.9)",
                            f"{(valid_icc >= 0.9).sum()}/{len(valid_icc)}"
                        )
                        col4.metric(
                            "Good (≥0.75)",
                            f"{(valid_icc >= 0.75).sum()}/{len(valid_icc)}"
                        )

                    # Download ICC results
                    csv = icc_results.to_csv(index=False)
                    st.download_button(
                        "Download ICC Results (CSV)",
                        data=csv,
                        file_name="icc_results.csv",
                        mime="text/csv",
                    )


def database_modeling_page():
    icon_heading(
        "Database & Modeling", "database", level=3,
        subtitle="Temporary cohort database, clinical follow-up import, and prediction modeling.",
    )

    tab_results, tab_clinical, tab_modeling = st.tabs([
        "Saved Results", "Clinical Follow-up", "Modeling",
    ])

    with tab_results:
        with section_card():
            icon_heading("Saved Omics Results", "database", level=4)
            st.caption(f"Temporary database path: `{DEFAULT_DB_PATH}`")
            saved = list_saved_results()
            if saved.empty:
                st.info("No saved omics results yet. Extract features in Beginner/Advanced mode and save them here.")
            else:
                st.dataframe(saved, use_container_width=True)
                wide = load_feature_wide_table()
                icon_heading("Modeling Feature Table", "file-spreadsheet", level=4, accent="secondary")
                st.dataframe(wide, use_container_width=True)
                st.download_button(
                    "Download Wide Feature Table (CSV)",
                    data=wide.to_csv(index=False),
                    file_name="radiomics_modeling_features.csv",
                    mime="text/csv",
                )

            if st.button("Clear Temporary Database", key='clear_temp_db'):
                clear_database()
                st.success("Temporary database cleared.")
                st.rerun()

    with tab_clinical:
        with section_card():
            icon_heading("Upload Clinical Follow-up Table", "upload", level=4)
            st.caption("Upload CSV/XLSX. One row per case is recommended.")
            clinical_file = st.file_uploader(
                "Clinical follow-up table",
                type=['csv', 'xlsx', 'xls'],
                key='clinical_upload',
            )
            if clinical_file is not None:
                clinical_df = read_tabular_upload(clinical_file)
                st.dataframe(clinical_df.head(30), use_container_width=True)
                case_id_col = st.selectbox(
                    "Column containing Case ID",
                    clinical_df.columns.tolist(),
                    key='clinical_case_id_col',
                )
                replace = st.checkbox(
                    "Replace existing clinical follow-up rows",
                    value=True,
                    key='clinical_replace',
                )
                if st.button("Save Clinical Follow-up", key='save_clinical_btn'):
                    rows = save_clinical_followup(
                        clinical_df, case_id_col=case_id_col,
                        source_name=getattr(clinical_file, 'name', 'clinical_followup'),
                        replace=replace,
                    )
                    st.success(f"Saved {rows} clinical follow-up rows.")

        with section_card():
            icon_heading("Current Clinical Follow-up", "clipboard-list", level=4, accent="secondary")
            current = load_clinical_followup()
            if current.empty:
                st.info("No clinical follow-up rows saved.")
            else:
                st.dataframe(current, use_container_width=True)

    with tab_modeling:
        modeling_df = build_modeling_table()
        with section_card():
            icon_heading("Modeling Dataset", "file-spreadsheet", level=4)
            if modeling_df.empty:
                st.info("No modeling data available. Save omics results and upload clinical follow-up first.")
                return
            st.dataframe(modeling_df.head(50), use_container_width=True)
            st.caption(f"{len(modeling_df)} cases × {len(modeling_df.columns)} columns")

        with section_card():
            icon_heading("Predictive Model Setup", "brain-circuit", level=4)
            with st.expander("How to use regression / classification modeling", expanded=True):
                st.markdown(
                    """
                    **Workflow**

                    1. In Beginner or Advanced mode, extract imaging/dose features for each case and click **Save Imaging Features** or **Save Dose Features**.
                    2. Open **Clinical Follow-up**, upload a CSV/XLSX table, and choose the column that matches the app's **Case ID**.
                    3. In this Modeling tab, choose an **Outcome / target column**. This is the value the model will predict.
                    4. Select **Omics predictors** and optional **Known clinical / follow-up predictors** as model inputs.
                    5. Choose **Task type**:
                       - **Auto**: numeric continuous outcomes are treated as regression; categorical or low-cardinality outcomes are treated as classification.
                       - **Regression**: use for continuous outcomes such as survival time, dose response, PSA value, toxicity score, or recurrence interval.
                       - **Classification**: use for grouped outcomes such as recurrence yes/no, toxicity grade group, responder/non-responder.
                    6. Set validation/test fractions. The remaining cases are used for model training.
                    7. Select one or more algorithms and click **Run Supervised Modeling**.

                    **Preprocessing**

                    - Numeric predictors can be mean/median imputed and optionally standardized.
                    - Categorical clinical columns are automatically one-hot encoded.
                    - Rows with missing target values are excluded from modeling.

                    **Outputs**

                    - Regression models report MAE, RMSE, and R2 on validation and test sets.
                    - Classification models report accuracy, balanced accuracy, F1, and ROC AUC when available.
                    - Use validation metrics for model selection and test metrics as the final held-out estimate.
                    """
                )
            target_options = [c for c in modeling_df.columns if c != 'case_id']
            if not target_options:
                st.warning("No target columns available.")
                return

            target_col = st.selectbox("Outcome / target column", target_options, key='model_target')
            omics_cols = [c for c in modeling_df.columns if '|' in c and c != target_col]
            clinical_cols = [
                c for c in modeling_df.columns
                if c not in {'case_id', target_col} and '|' not in c
            ]

            selected_omics = st.multiselect(
                "Omics predictors",
                omics_cols,
                default=omics_cols,
                key='model_omics_cols',
            )
            selected_clinical = st.multiselect(
                "Known clinical / follow-up predictors",
                clinical_cols,
                default=[],
                key='model_clinical_cols',
            )
            feature_cols = selected_omics + selected_clinical

            task_choice = st.selectbox(
                "Task type",
                ["Auto", "Regression", "Classification"],
                key='model_task_type',
            )
            inferred = infer_task_type(modeling_df[target_col])
            task_type = inferred if task_choice == "Auto" else task_choice.lower()
            st.caption(f"Using task type: `{task_type}`")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                validation_size = st.slider("Validation fraction", 0.1, 0.4, 0.2, 0.05)
            with col_b:
                test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
            with col_c:
                random_state = st.number_input("Random seed", value=42, step=1)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                numeric_imputer = st.selectbox("Numeric missing values", ["median", "mean"], key='model_imputer')
            with col_p2:
                scale_numeric = st.checkbox("Scale numeric predictors", value=True, key='model_scale')

            model_options = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
            selected_models = st.multiselect(
                "Models",
                model_options,
                default=model_options[:4],
                key='selected_models',
            )

            if st.button("Run Supervised Modeling", key='run_supervised_modeling'):
                if not feature_cols:
                    st.error("Select at least one predictor.")
                elif not selected_models:
                    st.error("Select at least one model.")
                else:
                    try:
                        results = run_supervised_models(
                            modeling_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            task_type=task_type,
                            model_names=selected_models,
                            validation_size=validation_size,
                            test_size=test_size,
                            random_state=int(random_state),
                            numeric_imputer=numeric_imputer,
                            scale_numeric=scale_numeric,
                        )
                        icon_heading("Model Evaluation", "bar-chart", level=4, accent="secondary")
                        st.caption(f"Split sizes: {results['split_sizes']}")
                        st.dataframe(results['metrics'], use_container_width=True)
                        st.download_button(
                            "Download Model Metrics (CSV)",
                            data=results['metrics'].to_csv(index=False),
                            file_name="model_metrics.csv",
                            mime="text/csv",
                        )
                    except ModuleNotFoundError:
                        st.error("Modeling requires scikit-learn. Install dependencies from requirements.txt.")
                    except Exception as e:
                        st.error(f"Modeling failed: {e}")

        with section_card():
            icon_heading("Unsupervised Clustering", "network", level=4)
            cluster_features = st.multiselect(
                "Features for clustering",
                [c for c in modeling_df.columns if c != 'case_id' and c != target_col],
                default=selected_omics[: min(20, len(selected_omics))],
                key='cluster_features',
            )
            n_clusters = st.slider("Number of clusters", 2, 8, 3, key='n_clusters')
            if st.button("Run KMeans Clustering", key='run_kmeans'):
                if not cluster_features:
                    st.error("Select at least one clustering feature.")
                else:
                    try:
                        results = run_kmeans_clustering(
                            modeling_df,
                            feature_cols=cluster_features,
                            n_clusters=n_clusters,
                            numeric_imputer=numeric_imputer,
                            scale_numeric=scale_numeric,
                        )
                        st.metric("Inertia", f"{results['metrics'].get('Inertia', np.nan):.3f}")
                        if 'Silhouette' in results['metrics']:
                            st.metric("Silhouette", f"{results['metrics']['Silhouette']:.3f}")
                        st.dataframe(results['assignments'], use_container_width=True)
                    except ModuleNotFoundError:
                        st.error("Clustering requires scikit-learn. Install dependencies from requirements.txt.")
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")


def main():
    # Header with logo
    st.markdown(get_header_html(), unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(get_sidebar_logo_html(), unsafe_allow_html=True)
    st.sidebar.markdown(get_sidebar_section_html("Mode", "sliders"), unsafe_allow_html=True)
    mode = st.sidebar.radio("Select mode", ["Beginner", "Advanced", "Database & Modeling"],
                            label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(get_sidebar_section_html("About", "info"), unsafe_allow_html=True)
    st.sidebar.markdown(
        "Open-source radiomics platform for medical image feature extraction. "
        "Supports CT/MR/PET with RTSTRUCT contours."
    )

    st.sidebar.markdown(get_sidebar_section_html("How to Cite", "book-open"), unsafe_allow_html=True)
    st.sidebar.markdown(
        "If you use this tool in your research, please cite:"
    )
    st.sidebar.caption("Radiomics Tools")
    st.sidebar.code(
        "Wang, X. Radiomics Tool: Medical Image\n"
        "Feature Extraction & Analysis Platform.\n"
        "https://github.com/wangxs89/Radiomics-Tools"
    )
    st.sidebar.caption("PyRadiomics")
    st.sidebar.code(
        "van Griethuysen, J. J. M., Fedorov, A., Parmar, C.,\n"
        "Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H.,\n"
        "Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017).\n"
        "Computational Radiomics System to Decode the Radiographic Phenotype.\n"
        "Cancer Research, 77(21), e104-e107.\n"
        "https://doi.org/10.1158/0008-5472.CAN-17-0339"
    )

    st.sidebar.markdown(get_sidebar_section_html("Contact", "mail"), unsafe_allow_html=True)
    st.sidebar.markdown(
        "For questions, bug reports, or feature requests:\n\n"
        "[wangxiaoshen0408@126.com]"
        "(mailto:wangxiaoshen0408@126.com)"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0 · Built with Streamlit & PyRadiomics")

    # Hero banner with radiomics illustration
    st.markdown(get_hero_banner_html(), unsafe_allow_html=True)

    if mode == "Beginner":
        beginner_mode()
    elif mode == "Advanced":
        advanced_mode()
    else:
        database_modeling_page()

    # Footer
    st.markdown(get_footer_html(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Beginner Mode
# ────────────────────────────────────────────

def beginner_mode():
    icon_heading(
        "Beginner Mode", "activity", level=3,
        subtitle="Select folder, verify ROI, extract features — three simple steps.",
    )

    init_state({
        'dicom_image': None, 'rois': None, 'roi_handler': None,
        'verification_complete': False, 'feature_extractor': None,
        'features_df': None, 'features_metadata': {},
    })

    current_step = 0
    if st.session_state.verification_complete:
        current_step = 2
    elif st.session_state.dicom_image is not None and st.session_state.rois is not None:
        current_step = 1

    st.markdown(get_step_indicator_html(["Select Data", "Verify ROI", "Extract"], current_step),
                unsafe_allow_html=True)

    # Step 0: Data loading
    with section_card():
        icon_heading("Select Data Folder", "folder-open", level=4)
        st.caption("Select a folder containing all DICOM files (images + RTSTRUCT)")
        data = folder_picker('')

    # Step 1: Visualization
    if (data['dicom_loaded'] and data['roi_loaded']
            and st.session_state.dicom_image is not None
            and st.session_state.rois is not None):
        verified = render_visualization(
            st.session_state.dicom_image, st.session_state.rois,
            st.session_state.roi_handler, key_prefix='',
        )
        if verified:
            st.session_state.verification_complete = True

    # Step 2: Combined Extraction (imaging + dose)
    if st.session_state.verification_complete:
        try:
            ensure_feature_extractor('feature_extractor')
        except Exception as e:
            st.error(f"Failed to initialize feature extractor: {e}")
            st.stop()

        rois = st.session_state.rois
        roi_names = st.session_state.roi_handler.get_roi_names(rois)
        dose_image = st.session_state.get('dose_image')
        ct_image = st.session_state.get('dicom_image')

        st.caption(f"Debug: {len(roi_names)} ROIs available, dose={'Yes' if dose_image else 'No'}")

        with section_card():
            icon_heading("Extract Features", "activity", level=4)

            selected_rois = []
            if not roi_names:
                st.warning("No ROIs available for extraction")
            else:
                selected_rois = st.multiselect(
                    "Select ROIs for feature extraction", roi_names,
                    default=roi_names[:1], key='feature_rois',
                )

            case_id = st.text_input(
                "Case ID",
                value=st.session_state.get('case_id') or get_default_case_id(
                    st.session_state.get('dicom_folder', '')
                ),
                key='case_id_input',
            ).strip()
            st.session_state['case_id'] = case_id

            imaging_series = st.session_state.get('imaging_series', [])
            current_series_index = st.session_state.get('selected_series_index', 0)
            selected_series_indices = [current_series_index]
            if len(imaging_series) > 1:
                selected_series_indices = st.multiselect(
                    "Imaging series to extract separately",
                    options=list(range(len(imaging_series))),
                    default=[current_series_index],
                    format_func=lambda i: make_series_label(imaging_series[i]),
                    key='batch_series_selector',
                )

            if st.button("Extract All Features"):
                progress = st.progress(0, text="Starting extraction...")
                try:
                    clear_extraction_results('')

                    # Step 1: Imaging features
                    progress.progress(10, text="Extracting imaging features...")
                    imaging_ok = run_imaging_batch_extraction(
                        st.session_state.feature_extractor,
                        rois, selected_rois,
                        imaging_series=imaging_series or [st.session_state.get('selected_series_info', {})],
                        selected_indices=selected_series_indices,
                        current_index=current_series_index,
                        current_image=ct_image,
                        case_id=case_id or "case",
                        key_prefix='',
                    )
                    progress.progress(50, text="Imaging features done")

                    # Step 2: Dose features (if dose available)
                    if dose_image is not None and imaging_ok:
                        try:
                            progress.progress(55, text="Creating ROI masks...")
                            dose_extractor = RadiomicsFeatureExtractor()
                            ct_masks = {}
                            for roi in rois:
                                if roi.name in selected_rois:
                                    try:
                                        m = dose_extractor.convert_roi_to_mask(roi, None, ct_image)
                                        if sitk.GetArrayFromImage(m).sum() > 0:
                                            ct_masks[roi.name] = m
                                    except Exception:
                                        pass

                            progress.progress(65, text="Resampling to dose grid...")
                            dose_masks = {}
                            for name, ct_m in ct_masks.items():
                                try:
                                    dm = resample_mask_to_image(ct_m, dose_image)
                                    if sitk.GetArrayFromImage(dm).sum() > 0:
                                        dose_masks[name] = dm
                                    else:
                                        try:
                                            dm2 = dose_extractor.convert_roi_to_mask(
                                                next(r for r in rois if r.name == name), None, dose_image)
                                            if sitk.GetArrayFromImage(dm2).sum() > 0:
                                                dose_masks[name] = dm2
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            progress.progress(75, text="Extracting dose features...")
                            if dose_masks:
                                df_dose = extract_dose_features(dose_image, dose_masks)

                                if df_dose.empty:
                                    st.warning("Dose masks were created, but no dose features could be extracted.")
                                else:
                                    st.session_state['dose_features_df'] = df_dose
                            else:
                                st.warning(
                                    "No valid dose masks overlapped the RTDOSE grid. "
                                    "Try selecting a different ROI."
                                )
                            progress.progress(100, text="All features extracted!")
                        except Exception as e:
                            progress.empty()
                            st.error(f"Dose analysis failed: {e}")
                    else:
                        progress.progress(100, text="Done!")

                    progress.empty()
                    dose_df = st.session_state.get('dose_features_df')
                    dose_rows = len(dose_df) if dose_df is not None else 0
                    has_results = (
                        st.session_state.get('features_df') is not None
                        or (dose_df is not None and not dose_df.empty)
                    )
                    st.success(
                        f"Imaging features: {'OK' if imaging_ok else 'skipped'}"
                        + (f" | Dose features: {dose_rows} ROIs" if dose_image else "")
                    )
                    if has_results:
                        st.info("Results are shown below. Use the download buttons to save CSV/Excel files.")
                    else:
                        st.warning("No result table was created. Check the messages above for the extraction error.")
                except Exception as e:
                    progress.empty()
                    st.error(f"Extraction failed: {e}")
                    st.exception(e)

    # Display cached imaging results
    if st.session_state.features_df is not None:
        render_results(
            st.session_state.features_df,
            st.session_state.features_metadata,
            key_prefix='',
        )

    # Display cached dose results
    dose_df = st.session_state.get('dose_features_df')
    if dose_df is not None and not dose_df.empty:
        with section_card():
            icon_heading("Dose Feature Matrix", "activity", level=4, accent="secondary")
            st.dataframe(dose_df, use_container_width=True)
            exporter = ResultsExporter()
            csv = exporter.to_csv(dose_df)
            st.download_button(
                label="Download Dose Features (CSV)", data=csv,
                file_name="dose_features.csv", mime="text/csv",
            )

    render_database_save_controls('')


# ─────────────────────────────────────────────
#  Advanced Mode
# ─────────────────────────────────────────────

def advanced_mode():
    icon_heading(
        "Advanced Mode", "settings", level=3,
        subtitle="Full control: preprocessing, filters, custom features.",
    )

    init_state({
        'adv_dicom_image': None, 'adv_rois': None, 'adv_roi_handler': None,
        'adv_verification_complete': False, 'adv_feature_extractor': None,
        'adv_preprocessed_image': None, 'adv_selected_filters': ['original'],
        'adv_feature_types': {
            'shape': True, 'firstorder': True, 'glcm': True,
            'gldm': True, 'glrlm': True, 'glszm': True, 'ngtdm': True,
            'shape2D': False,
        },
        'adv_features_df': None, 'adv_features_metadata': {},
    })

    # Determine current step
    adv_step = 0
    if st.session_state.adv_verification_complete:
        adv_step = 5
    elif st.session_state.adv_dicom_image is not None and st.session_state.adv_rois is not None:
        adv_step = 4
    elif st.session_state.adv_dicom_image is not None:
        adv_step = 1

    st.markdown(
        get_step_indicator_html(
            ["Data", "Preprocess", "Filters", "Features", "Verify", "Extract"], adv_step,
        ),
        unsafe_allow_html=True,
    )

    # Step 0: Data loading
    with section_card():
        icon_heading("Select Data Folder", "folder-open", level=4)
        data = folder_picker('adv_')

    # Step 1: Preprocessing
    with section_card():
        icon_heading("Image Preprocessing", "sliders", level=4)
        original_image = st.session_state.adv_dicom_image
        processed_image = original_image

        preprocess_enabled = st.checkbox("Enable preprocessing", value=False)
        if preprocess_enabled and original_image is not None:
            preprocessor = ImagePreprocessor()

            col1, col2 = st.columns(2)
            with col1:
                icon_heading("Resampling", "scan", level=5, accent="secondary")
                if st.checkbox("Enable resampling", key='adv_resample_en'):
                    sx = st.number_input("X spacing (mm)", value=1.0, step=0.1, key='adv_res_x')
                    sy = st.number_input("Y spacing (mm)", value=1.0, step=0.1, key='adv_res_y')
                    sz = st.number_input("Z spacing (mm)", value=1.0, step=0.1, key='adv_res_z')
                    if st.button("Apply Resampling", key='adv_resample_btn'):
                        processed_image = preprocessor.resample_image(
                            processed_image, [sx, sy, sz],
                        )
                        st.success("Resampling complete")

            with col2:
                icon_heading("Normalization", "sliders", level=5, accent="secondary")
                method = st.selectbox(
                    "Method", ["None", "z-score", "min-max", "percentile"], key='adv_norm_method',
                )
                if method != "None" and st.button("Apply Normalization", key='adv_norm_btn'):
                    processed_image = preprocessor.normalize_image(processed_image, method=method)
                    st.success(f"{method} normalization complete")

            icon_heading("Discretization", "grid", level=5, accent="secondary")
            if st.checkbox("Enable discretization", key='adv_disc_en'):
                bin_width = st.number_input("Bin width", value=25.0, step=1.0, key='adv_disc_bw')
                if st.button("Apply Discretization", key='adv_disc_btn'):
                    processed_image = preprocessor.discretize_image(processed_image, bin_width=bin_width)
                    st.success("Discretization complete")

            if processed_image is not original_image:
                st.session_state.adv_preprocessed_image = processed_image
                st.info("Preprocessed image saved for feature extraction")

    # Step 2: Filter selection
    with section_card():
        icon_heading("Filter Selection", "filter", level=4)
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            icon_heading("Available Filters", "filter", level=5, accent="secondary")
            filter_options = {k: v['name'] for k, v in RadiomicsFilterConfig.AVAILABLE_FILTERS.items()}
            filter_labels = list(filter_options.values())
            selected_labels = st.multiselect(
                "Select filters", filter_labels,
                default=[filter_options['original']], key='adv_filter_sel',
            )
            selected_filters = []
            for label in selected_labels:
                for key, val in filter_options.items():
                    if val == label:
                        selected_filters.append(key)
                        break

        with col_filter2:
            icon_heading("Filter Parameters", "settings", level=5, accent="secondary")
            log_sigmas = None
            if 'log' in selected_filters:
                log_input = st.text_input(
                    "LoG Sigma values (comma-separated)",
                    value="1.0,2.0,3.0", key='adv_log_sigmas',
                )
                log_sigmas = [float(s.strip()) for s in log_input.split(',') if s.strip()]

            wavelet_types = None
            if 'wavelet' in selected_filters:
                wavelet_types = st.multiselect(
                    "Wavelet types",
                    ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH'],
                    default=['LLL', 'LLH', 'LHL', 'LHH'], key='adv_wavelet',
                )

            icon_heading("PyRadiomics Settings", "sliders", level=5, accent="secondary")
            pyrad_bin_width = st.number_input(
                "Gray-level bin width",
                min_value=0.1, value=25.0, step=1.0, key='adv_pyrad_binwidth',
            )
            use_pyrad_resampling = st.checkbox(
                "Resample inside PyRadiomics", value=False, key='adv_pyrad_resample_en',
            )
            pyrad_resampled_spacing = None
            if use_pyrad_resampling:
                rx = st.number_input("PyRadiomics X spacing (mm)", min_value=0.1, value=1.0, step=0.1, key='adv_pyrad_res_x')
                ry = st.number_input("PyRadiomics Y spacing (mm)", min_value=0.1, value=1.0, step=0.1, key='adv_pyrad_res_y')
                rz = st.number_input("PyRadiomics Z spacing (mm)", min_value=0.1, value=1.0, step=0.1, key='adv_pyrad_res_z')
                pyrad_resampled_spacing = [rx, ry, rz]
            pyrad_force2d = st.checkbox(
                "Force 2D extraction", value=False, key='adv_force2d_en',
            )
            pyrad_force2d_dimension = st.selectbox(
                "Force 2D dimension", [0, 1, 2], index=0, key='adv_force2d_dim',
            )

    # Step 3: Feature types
    with section_card():
        icon_heading("Feature Types", "grid", level=4)
        col_feat1, col_feat2, col_feat3 = st.columns(3)

        with col_feat1:
            shape_enabled = st.checkbox("Shape", value=True, key='adv_feat_shape')
            firstorder_enabled = st.checkbox("First Order", value=True, key='adv_feat_fo')
            glcm_enabled = st.checkbox("GLCM", value=True, key='adv_feat_glcm')
        with col_feat2:
            gldm_enabled = st.checkbox("GLDM", value=True, key='adv_feat_gldm')
            glrlm_enabled = st.checkbox("GLRLM", value=True, key='adv_feat_glrlm')
            glszm_enabled = st.checkbox("GLSZM", value=True, key='adv_feat_glszm')
        with col_feat3:
            ngtdm_enabled = st.checkbox("NGTDM", value=True, key='adv_feat_ngtdm')
            shape2d_enabled = st.checkbox("Shape 2D", value=False, key='adv_feat_shape2d')

        feature_classes = {
            'shape': shape_enabled, 'firstorder': firstorder_enabled,
            'glcm': glcm_enabled, 'gldm': gldm_enabled,
            'glrlm': glrlm_enabled, 'glszm': glszm_enabled,
            'ngtdm': ngtdm_enabled, 'shape2D': shape2d_enabled,
        }
        enabled_count = sum(1 for v in feature_classes.values() if v)
        st.caption(f"{enabled_count} feature classes enabled")
        if shape2d_enabled and not pyrad_force2d:
            st.caption("Shape 2D selected: force2D will be enabled automatically.")

    filter_settings = RadiomicsFilterConfig.build_settings(
        enabled_filters=selected_filters,
        log_sigmas=log_sigmas,
        wavelet_types=wavelet_types,
        bin_width=pyrad_bin_width,
        resampled_pixel_spacing=pyrad_resampled_spacing,
        force_2d=pyrad_force2d or shape2d_enabled,
        force_2d_dimension=pyrad_force2d_dimension,
    )
    st.caption(f"{len(filter_settings.get('enabledImageTypes', []))} image types configured")

    # Step 4: Visualization
    image_for_viz = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image
    if (data['dicom_loaded'] and data['roi_loaded']
            and image_for_viz is not None
            and st.session_state.adv_rois is not None):
        verified = render_visualization(
            image_for_viz, st.session_state.adv_rois,
            st.session_state.adv_roi_handler, key_prefix='adv_',
        )
        if verified:
            st.session_state.adv_verification_complete = True

    # Step 5: Combined Extraction (imaging + dose)
    if st.session_state.adv_verification_complete:
        ensure_feature_extractor('adv_feature_extractor')

        rois = st.session_state.adv_rois
        roi_names = st.session_state.adv_roi_handler.get_roi_names(rois)
        dose_image = st.session_state.get('adv_dose_image')
        ct_image = st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image

        with section_card():
            icon_heading("Extract Features", "activity", level=4)
            selected_rois = st.multiselect(
                "Select ROIs for feature extraction", roi_names,
                default=roi_names[:1], key='adv_feature_rois',
            )

            case_id = st.text_input(
                "Case ID",
                value=st.session_state.get('adv_case_id') or get_default_case_id(
                    st.session_state.get('adv_dicom_folder', '')
                ),
                key='adv_case_id_input',
            ).strip()
            st.session_state['adv_case_id'] = case_id

            imaging_series = st.session_state.get('adv_imaging_series', [])
            current_series_index = st.session_state.get('adv_selected_series_index', 0)
            selected_series_indices = [current_series_index]
            if len(imaging_series) > 1:
                selected_series_indices = st.multiselect(
                    "Imaging series to extract separately",
                    options=list(range(len(imaging_series))),
                    default=[current_series_index],
                    format_func=lambda i: make_series_label(imaging_series[i]),
                    key='adv_batch_series_selector',
                )

            if st.button("Extract All Features", key='adv_extract_btn'):
                progress = st.progress(0, text="Starting extraction...")
                clear_extraction_results('adv_')

                # Step 1: Imaging features
                progress.progress(10, text="Extracting imaging features...")
                imaging_ok = run_imaging_batch_extraction(
                    st.session_state.adv_feature_extractor,
                    rois, selected_rois,
                    imaging_series=imaging_series or [st.session_state.get('adv_selected_series_info', {})],
                    selected_indices=selected_series_indices,
                    current_index=current_series_index,
                    current_image=ct_image,
                    case_id=case_id or "case",
                    key_prefix='adv_',
                    feature_classes=feature_classes,
                    filter_settings=filter_settings,
                )
                progress.progress(50, text="Imaging features done")

                # Step 2: Dose features
                if dose_image is not None and imaging_ok:
                    try:
                        progress.progress(55, text="Creating ROI masks...")
                        dose_extractor = RadiomicsFeatureExtractor()
                        ct_masks = {}
                        for roi in rois:
                            if roi.name in selected_rois:
                                try:
                                    m = dose_extractor.convert_roi_to_mask(roi, None, st.session_state.adv_dicom_image)
                                    if sitk.GetArrayFromImage(m).sum() > 0:
                                        ct_masks[roi.name] = m
                                except Exception:
                                    pass

                        progress.progress(65, text="Resampling to dose grid...")
                        dose_masks = {}
                        for name, ct_m in ct_masks.items():
                            try:
                                dm = resample_mask_to_image(ct_m, dose_image)
                                if sitk.GetArrayFromImage(dm).sum() > 0:
                                    dose_masks[name] = dm
                                else:
                                    try:
                                        dm2 = dose_extractor.convert_roi_to_mask(
                                            next(r for r in rois if r.name == name), None, dose_image)
                                        if sitk.GetArrayFromImage(dm2).sum() > 0:
                                            dose_masks[name] = dm2
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                        progress.progress(75, text="Extracting dose features...")
                        if dose_masks:
                            df_dose = extract_dose_features(dose_image, dose_masks)

                            if df_dose.empty:
                                st.warning("Dose masks were created, but no dose features could be extracted.")
                            else:
                                st.session_state['adv_dose_features_df'] = df_dose
                        else:
                            st.warning(
                                "No valid dose masks overlapped the RTDOSE grid. "
                                "Try selecting a different ROI."
                            )
                        progress.progress(100, text="All features extracted!")
                    except Exception as e:
                        progress.empty()
                        st.error(f"Dose analysis failed: {e}")
                else:
                    progress.progress(100, text="Done!")

                progress.empty()
                dose_df = st.session_state.get('adv_dose_features_df')
                dose_rows = len(dose_df) if dose_df is not None else 0
                has_results = (
                    st.session_state.get('adv_features_df') is not None
                    or (dose_df is not None and not dose_df.empty)
                )
                st.success(
                    f"Imaging features: {'OK' if imaging_ok else 'skipped'}"
                    + (f" | Dose features: {dose_rows} ROIs" if dose_image else "")
                )
                if has_results:
                    st.info("Results are shown below. Use the download buttons to save CSV/Excel files.")
                else:
                    st.warning("No result table was created. Check the messages above for the extraction error.")

    # Display cached imaging results
    if st.session_state.adv_features_df is not None:
        render_results(
            st.session_state.adv_features_df,
            st.session_state.adv_features_metadata,
            key_prefix='adv_', show_report=True,
        )

    # Display cached dose results
    dose_df = st.session_state.get('adv_dose_features_df')
    if dose_df is not None and not dose_df.empty:
        with section_card():
            icon_heading("Dose Feature Matrix", "activity", level=4, accent="secondary")
            st.dataframe(dose_df, use_container_width=True)
            exporter = ResultsExporter()
            csv = exporter.to_csv(dose_df)
            st.download_button(
                label="Download Dose Features (CSV)", data=csv,
                file_name="dose_features_advanced.csv", mime="text/csv",
            )

    render_database_save_controls('adv_')


if __name__ == "__main__":
    main()
