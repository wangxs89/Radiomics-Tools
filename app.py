"""Radiomics Web App — Main Application"""
from contextlib import contextmanager
from typing import Optional
import subprocess
import streamlit as st
from pathlib import Path
import numpy as np
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
    load_dose_image, compute_dth, compute_ovh, compute_dvh,
    find_ptv_rois, extract_dose_features, resample_mask_to_image,
)
from src.ui_theme import APP_CSS, get_step_indicator_html, get_header_html, get_sidebar_logo_html, get_hero_banner_html, get_footer_html


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


@contextmanager
def section_card():
    """Context manager that wraps content in a styled section card."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown('</div>', unsafe_allow_html=True)


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
            series_info.append({
                'id': sid, 'files': file_names,
                'modality': modality, 'description': series_desc,
                'count': len(file_names),
            })

    rtstruct_files = [f for f, m in file_modalities.items() if m == 'RTSTRUCT']
    imaging_series = [s for s in series_info
                      if s['modality'] in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR')]
    dose_series = [s for s in series_info if s['modality'] == 'RTDOSE']
    other_series = [s for s in series_info
                    if s['modality'] not in ('CT', 'MR', 'PT', 'MG', 'US', 'XA', 'DX', 'CR', 'RTDOSE', 'RTSTRUCT')]

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
        st.success(
            f"Loaded: {selected['modality']} - {selected['description'] or 'Unnamed'} "
            f"({sitk_image.GetSize()[0]}×{sitk_image.GetSize()[1]}×{sitk_image.GetSize()[2]})"
        )
        st.write(f"**HU range:** {max(arr.min(), -1024):.0f} ~ {arr.max():.0f}")
    else:
        st.warning("No imaging series found (CT/MR/PET etc.)")

    # Dose loading
    dose_image = None
    if dose_series:
        st.markdown("**Dose Files (RTDOSE):**")
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
        st.markdown("#### Verify ROI Placement")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Select ROIs to display**")
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

            st.markdown("**Window / Level**")
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

def run_extraction(feature_extractor, sitk_image, rois, selected_rois,
                   key_prefix: str = '', **extract_kwargs) -> bool:
    """Run feature extraction, save results to session_state. Returns success bool."""
    try:
        masks_dict = {}
        for roi in rois:
            if roi.name in selected_rois:
                mask = feature_extractor.convert_roi_to_mask(roi, None, sitk_image)
                masks_dict[roi.name] = mask

        if not masks_dict:
            st.warning("No ROIs selected for extraction")
            return False

        df_features = feature_extractor.extract_features_for_rois(
            sitk_image, masks_dict, **extract_kwargs,
        )
        if df_features.empty:
            st.warning("No features could be extracted")
            return False

        st.session_state[f'{key_prefix}features_df'] = df_features
        st.session_state[f'{key_prefix}features_metadata'] = {
            'Image size': f"{sitk_image.GetSize()[0]}×{sitk_image.GetSize()[1]}×{sitk_image.GetSize()[2]}",
            'Voxel spacing': (
                f"{sitk_image.GetSpacing()[0]:.2f}×"
                f"{sitk_image.GetSpacing()[1]:.2f}×"
                f"{sitk_image.GetSpacing()[2]:.2f} mm"
            ),
            'ROI count': len(masks_dict),
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
        st.subheader("Feature Matrix")
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

        st.subheader("Feature Summary Statistics")
        summary_stats = exporter.get_summary_stats(df_features)
        st.dataframe(summary_stats, use_container_width=True)

    if show_report:
        render_visualization_report(df_features, key_prefix)


def render_visualization_report(df_features, key_prefix: str = '') -> None:
    """Render the 4-tab visualization report (advanced mode only)."""
    with section_card():
        st.markdown("---")
        st.subheader("Visualization Report")

        report_gen = ReportGenerator()

        tab1, tab2, tab3, tab4 = st.tabs([
            "Correlation Heatmap", "Feature Distribution", "Box Plot", "Summary Statistics",
        ])

        with tab1:
            st.markdown("**Feature Correlation Heatmap**")
            try:
                fig = report_gen.create_correlation_heatmap(df_features)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {e}")

        with tab2:
            st.markdown("**Feature Distribution Histogram**")
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
            st.markdown("**ROI Comparison Box Plot**")
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
            st.markdown("**Detailed Summary Statistics**")
            try:
                summary = report_gen.create_summary_table(df_features)
                st.dataframe(summary, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate summary statistics: {e}")


# ─────────────────────────────────────────────
#  Dosomics Section
# ─────────────────────────────────────────────

def render_dosomics_section(ct_image, dose_image, rois, handler, key_prefix: str = ''):
    """Render dose feature extraction and DTH/OVH analysis.

    Args:
        ct_image: CT SimpleITK image (used for initial mask creation)
        dose_image: Dose SimpleITK image (target grid for features)
        rois: list of ROI objects
        handler: ROIHandler instance
        key_prefix: session state key prefix
    """
    with section_card():
        st.markdown("#### Dose Analysis")
        st.caption("Extract features from dose distribution and analyze OAR-target spatial relationships")

        roi_names = handler.get_roi_names(rois)
        ptv_candidates = find_ptv_rois(roi_names)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Target Volume**")
            selected_ptv = st.selectbox(
                "Select PTV/Target",
                ptv_candidates if ptv_candidates else roi_names,
                key=f'{key_prefix}ptv_select',
            )
        with col2:
            st.markdown("**OARs to Analyze**")
            oar_options = [n for n in roi_names if n != selected_ptv]
            selected_oars = st.multiselect(
                "Select OARs",
                oar_options,
                default=oar_options[:3] if len(oar_options) > 3 else oar_options,
                key=f'{key_prefix}oar_select',
            )

        if st.button("Extract Dose Features & Compute DTH/OVH", key=f'{key_prefix}dose_extract_btn'):
            progress = st.progress(0, text="Starting dose analysis...")

            try:
                from radiomics import featureextractor
                import plotly.graph_objects as go

                # Step 1: Create ROI masks on CT grid, then resample to dose grid
                progress.progress(10, text="Creating ROI masks on CT grid...")
                dose_extractor = RadiomicsFeatureExtractor()
                target_names = [selected_ptv] + selected_oars
                ct_masks_dict = {}

                for roi in rois:
                    if roi.name in target_names:
                        try:
                            ct_mask = dose_extractor.convert_roi_to_mask(roi, None, ct_image)
                            ct_mask_arr = sitk.GetArrayFromImage(ct_mask)
                            if ct_mask_arr.sum() > 0:
                                ct_masks_dict[roi.name] = ct_mask
                        except Exception:
                            pass

                progress.progress(25, text=f"Created {len(ct_masks_dict)} CT masks, resampling to dose grid...")

                # Resample CT masks to dose grid
                masks_dict = {}
                for roi_name, ct_mask in ct_masks_dict.items():
                    try:
                        dose_mask = resample_mask_to_image(ct_mask, dose_image)
                        dose_mask_arr = sitk.GetArrayFromImage(dose_mask)
                        if dose_mask_arr.sum() > 0:
                            masks_dict[roi_name] = dose_mask
                        else:
                            # Fallback: try creating mask directly on dose grid
                            try:
                                direct_mask = dose_extractor.convert_roi_to_mask(
                                    next(r for r in rois if r.name == roi_name), None, dose_image,
                                )
                                direct_arr = sitk.GetArrayFromImage(direct_mask)
                                if direct_arr.sum() > 0:
                                    masks_dict[roi_name] = direct_mask
                            except Exception:
                                pass
                    except Exception:
                        pass

                progress.progress(30, text=f"Created {len(masks_dict)} ROI masks")

                if not masks_dict:
                    st.error("No valid ROI masks could be created on the dose grid. "
                             "The dose image geometry may not overlap with the ROI contours.")
                    progress.empty()
                    return

                # Step 2: Extract dose features
                progress.progress(40, text="Extracting dose features...")
                df_dose = extract_dose_features(dose_image, masks_dict)
                st.session_state[f'{key_prefix}dose_features_df'] = df_dose
                progress.progress(60, text=f"Extracted features for {len(df_dose)} ROIs")

                # Step 3: Compute DTH/OVH
                progress.progress(70, text="Computing DTH/OVH curves...")
                dth_results = {}
                ovh_results = {}
                ptv_mask = masks_dict.get(selected_ptv)

                if ptv_mask is not None:
                    ptv_arr = sitk.GetArrayFromImage(ptv_mask)
                    spacing = dose_image.GetSpacing()
                    total_oars = len(selected_oars)

                    for i, oar_name in enumerate(selected_oars):
                        oar_mask = masks_dict.get(oar_name)
                        if oar_mask is not None:
                            oar_arr = sitk.GetArrayFromImage(oar_mask)
                            dth_bins, dth_fracs = compute_dth(oar_arr, ptv_arr, spacing)
                            ovh_dists, ovh_fracs = compute_ovh(oar_arr, ptv_arr, spacing)
                            dth_results[oar_name] = (dth_bins, dth_fracs)
                            ovh_results[oar_name] = (ovh_dists, ovh_fracs)

                        pct = 70 + int(25 * (i + 1) / max(total_oars, 1))
                        progress.progress(pct, text=f"DTH/OVH: {oar_name}")
                else:
                    st.warning(f"PTV mask '{selected_ptv}' not available for DTH/OVH computation")

                st.session_state[f'{key_prefix}dth_results'] = dth_results
                st.session_state[f'{key_prefix}ovh_results'] = ovh_results
                progress.progress(100, text="Dose analysis complete!")
                st.success(f"Dose features extracted for {len(masks_dict)} ROIs, "
                           f"DTH/OVH computed for {len(dth_results)} OARs")
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"Dose analysis failed: {e}")
                st.exception(e)

    # Display cached dose results
    dose_df = st.session_state.get(f'{key_prefix}dose_features_df')
    if dose_df is not None and not dose_df.empty:
        with section_card():
            st.markdown("#### Dose Feature Matrix")
            st.dataframe(dose_df, use_container_width=True)

            exporter = ResultsExporter()
            csv = exporter.to_csv(dose_df)
            st.download_button(
                label="Download Dose Features (CSV)",
                data=csv,
                file_name=f"dose_features_{key_prefix or 'beginner'}.csv",
                mime="text/csv",
            )

        # DTH/OVH Visualization
        dth_results = st.session_state.get(f'{key_prefix}dth_results', {})
        ovh_results = st.session_state.get(f'{key_prefix}ovh_results', {})

        if dth_results or ovh_results:
            with section_card():
                st.markdown("#### Spatial Relationship Analysis")
                st.caption("DTH = Distance-to-Target Histogram | OVH = Overlap Volume Histogram")

                import plotly.graph_objects as go

                dth_tab, ovh_tab = st.tabs(["DTH Curves", "OVH Curves"])

                with dth_tab:
                    if dth_results:
                        fig_dth = go.Figure()
                        for oar_name, (bins, fracs) in dth_results.items():
                            fig_dth.add_trace(go.Scatter(
                                x=bins, y=fracs * 100, mode='lines+markers',
                                name=oar_name, marker_size=4,
                            ))
                        fig_dth.update_layout(
                            title="Distance-to-Target Histogram (DTH)",
                            xaxis_title="Distance to Target (mm)",
                            yaxis_title="OAR Volume Fraction (%)",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                            margin=dict(t=40, b=80),
                        )
                        st.plotly_chart(fig_dth, use_container_width=True)
                    else:
                        st.info("No DTH data available. Select OARs and re-run extraction.")

                with ovh_tab:
                    if ovh_results:
                        fig_ovh = go.Figure()
                        for oar_name, (dists, fracs) in ovh_results.items():
                            fig_ovh.add_trace(go.Scatter(
                                x=dists, y=fracs * 100, mode='lines+markers',
                                name=oar_name, marker_size=4,
                            ))
                        fig_ovh.update_layout(
                            title="Overlap Volume Histogram (OVH)",
                            xaxis_title="Distance from Target Surface (mm)",
                            yaxis_title="Cumulative OAR Overlap (%)",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                            margin=dict(t=40, b=80),
                        )
                        st.plotly_chart(fig_ovh, use_container_width=True)
                    else:
                        st.info("No OVH data available. Select OARs and re-run extraction.")


# ─────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────

def main():
    # Header with logo
    st.markdown(get_header_html(), unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(get_sidebar_logo_html(), unsafe_allow_html=True)
    st.sidebar.markdown("### Mode")
    mode = st.sidebar.radio("Select mode", ["Beginner", "Advanced"],
                            label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Open-source radiomics platform for medical image feature extraction. "
        "Supports CT/MR/PET with RTSTRUCT contours."
    )

    st.sidebar.markdown("### How to Cite")
    st.sidebar.markdown(
        "If you use this tool in your research, please cite:"
    )
    st.sidebar.code(
        "Wang, X. Radiomics Tool: Medical Image\n"
        "Feature Extraction & Analysis Platform.\n"
        "https://github.com/wangxs89/Radiomics-Tools"
    )

    st.sidebar.markdown("### Contact")
    st.sidebar.markdown(
        "For questions, bug reports, or feature requests:\n\n"
        "📧 [wangxiaoshen0408@126.com]"
        "(mailto:wangxiaoshen0408@126.com)"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0 · Built with Streamlit & PyRadiomics")

    # Hero banner with radiomics illustration
    st.markdown(get_hero_banner_html(), unsafe_allow_html=True)

    if mode == "Beginner":
        beginner_mode()
    else:
        advanced_mode()

    # Footer
    st.markdown(get_footer_html(), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Beginner Mode
# ────────────────────────────────────────────

def beginner_mode():
    st.markdown("### Beginner Mode")
    st.caption("Select folder, verify ROI, extract features — three simple steps.")

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
        st.markdown("#### Select Data Folder")
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

    # Step 2: Extraction
    if st.session_state.verification_complete:
        init_state({'feature_extractor': RadiomicsFeatureExtractor()})

        rois = st.session_state.rois
        roi_names = st.session_state.roi_handler.get_roi_names(rois)

        with section_card():
            st.markdown("#### 2. Extract Features")
            selected_rois = st.multiselect(
                "Select ROIs for feature extraction", roi_names,
                default=roi_names[:1], key='feature_rois',
            )

            if st.button("Extract Features"):
                run_extraction(
                    st.session_state.feature_extractor,
                    st.session_state.dicom_image,
                    rois, selected_rois, key_prefix='',
                )
                st.rerun()

    # Display cached results
    if st.session_state.features_df is not None:
        render_results(
            st.session_state.features_df,
            st.session_state.features_metadata,
            key_prefix='',
        )

    # Dosomics section
    dose_image = st.session_state.get('dose_image')
    ct_image = st.session_state.get('dicom_image')
    if dose_image is not None and ct_image is not None and st.session_state.rois is not None:
        render_dosomics_section(
            ct_image, dose_image, st.session_state.rois, st.session_state.roi_handler,
            key_prefix='',
        )


# ─────────────────────────────────────────────
#  Advanced Mode
# ─────────────────────────────────────────────

def advanced_mode():
    st.markdown("### Advanced Mode")
    st.caption("Full control: preprocessing, filters, custom features.")

    init_state({
        'adv_dicom_image': None, 'adv_rois': None, 'adv_roi_handler': None,
        'adv_verification_complete': False, 'adv_feature_extractor': None,
        'adv_preprocessed_image': None, 'adv_selected_filters': ['original'],
        'adv_feature_types': {
            'shape': True, 'firstorder': True, 'glcm': True,
            'gldm': True, 'glrlm': True, 'glszm': True, 'ngtdm': True,
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
        st.markdown("#### Select Data Folder")
        data = folder_picker('adv_')

    # Step 1: Preprocessing
    with section_card():
        st.markdown("#### 1. Image Preprocessing")
        original_image = st.session_state.adv_dicom_image
        processed_image = original_image

        preprocess_enabled = st.checkbox("Enable preprocessing", value=False)
        if preprocess_enabled and original_image is not None:
            preprocessor = ImagePreprocessor()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Resampling**")
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
                st.markdown("**Normalization**")
                method = st.selectbox(
                    "Method", ["None", "z-score", "min-max", "percentile"], key='adv_norm_method',
                )
                if method != "None" and st.button("Apply Normalization", key='adv_norm_btn'):
                    processed_image = preprocessor.normalize_image(processed_image, method=method)
                    st.success(f"{method} normalization complete")

            st.markdown("**Discretization**")
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
        st.markdown("#### 2. Filter Selection")
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            st.markdown("**Available Filters**")
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
            st.markdown("**Filter Parameters**")
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

        filter_settings = RadiomicsFilterConfig.build_settings(
            enabled_filters=selected_filters,
            log_sigmas=log_sigmas,
            wavelet_types=wavelet_types,
        )
        st.caption(f"{len(filter_settings.get('enabledImageTypes', []))} image types configured")

    # Step 3: Feature types
    with section_card():
        st.markdown("#### 3. Feature Types")
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

        feature_classes = {
            'shape': shape_enabled, 'firstorder': firstorder_enabled,
            'glcm': glcm_enabled, 'gldm': gldm_enabled,
            'glrlm': glrlm_enabled, 'glszm': glszm_enabled,
            'ngtdm': ngtdm_enabled,
        }
        enabled_count = sum(1 for v in feature_classes.values() if v)
        st.caption(f"{enabled_count} feature classes enabled")

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

    # Step 5: Extraction
    if st.session_state.adv_verification_complete:
        init_state({'adv_feature_extractor': RadiomicsFeatureExtractor()})

        rois = st.session_state.adv_rois
        roi_names = st.session_state.adv_roi_handler.get_roi_names(rois)

        with section_card():
            st.markdown("#### 5. Extract Features")
            selected_rois = st.multiselect(
                "Select ROIs for feature extraction", roi_names,
                default=roi_names[:1], key='adv_feature_rois',
            )

            if st.button("Extract Features", key='adv_extract_btn'):
                run_extraction(
                    st.session_state.adv_feature_extractor,
                    st.session_state.adv_preprocessed_image or st.session_state.adv_dicom_image,
                    rois, selected_rois, key_prefix='adv_',
                    feature_classes=feature_classes,
                    filter_settings=filter_settings,
                )
                st.rerun()

    # Display cached results
    if st.session_state.adv_features_df is not None:
        render_results(
            st.session_state.adv_features_df,
            st.session_state.adv_features_metadata,
            key_prefix='adv_', show_report=True,
        )

    # Dosomics section
    dose_image = st.session_state.get('adv_dose_image')
    ct_image = st.session_state.get('adv_dicom_image')
    if dose_image is not None and ct_image is not None and st.session_state.adv_rois is not None:
        render_dosomics_section(
            ct_image, dose_image, st.session_state.adv_rois, st.session_state.adv_roi_handler,
            key_prefix='adv_',
        )


if __name__ == "__main__":
    main()
