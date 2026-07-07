import pandas as pd

from src.data_store import (
    build_modeling_table,
    list_saved_results,
    load_feature_wide_table,
    save_clinical_followup,
    save_feature_dataframe,
)


def test_feature_results_save_and_widen(tmp_path):
    db_path = tmp_path / "radiomics.sqlite3"
    features = pd.DataFrame({
        "CaseID": ["case-1"],
        "Series": ["CT - A (10 slices)"],
        "SeriesUID": ["1.2.3"],
        "Modality": ["CT"],
        "ROI": ["PTV"],
        "original_firstorder_Mean": [12.5],
        "original_shape_VoxelVolume": [42.0],
    })

    written = save_feature_dataframe(
        features, case_id="case-1", feature_kind="imaging", db_path=db_path,
    )
    saved = list_saved_results(db_path)
    wide = load_feature_wide_table(db_path)

    assert written == 1
    assert len(saved) == 1
    assert wide["case_id"].iloc[0] == "case-1"
    assert "imaging|CT - A (10 slices)|PTV|original_firstorder_Mean" in wide.columns


def test_modeling_table_merges_clinical_followup(tmp_path):
    db_path = tmp_path / "radiomics.sqlite3"
    features = pd.DataFrame({
        "ROI": ["PTV"],
        "Series": ["CT"],
        "original_firstorder_Mean": [7.0],
    })
    clinical = pd.DataFrame({
        "patient_id": ["case-1"],
        "age": [65],
        "followup_score": [0.8],
    })

    save_feature_dataframe(
        features, case_id="case-1", feature_kind="imaging", db_path=db_path,
    )
    save_clinical_followup(
        clinical, case_id_col="patient_id", db_path=db_path,
    )
    modeling = build_modeling_table(db_path)

    assert len(modeling) == 1
    assert modeling["case_id"].iloc[0] == "case-1"
    assert modeling["age"].iloc[0] == 65
    assert "imaging|CT|PTV|original_firstorder_Mean" in modeling.columns
