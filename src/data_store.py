"""Local temporary database for radiomics case-level results."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


DB_DIR = Path(".radiomics_tmp")
DEFAULT_DB_PATH = DB_DIR / "radiomics_results.sqlite3"

FEATURE_META_COLUMNS = {
    "ROI", "CaseID", "Case ID", "case_id",
    "SeriesUID", "Series", "SeriesLabel", "SeriesDescription",
    "Modality", "FeatureKind", "feature_kind",
}


def get_db_path(db_path: Optional[Path | str] = None) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            case_id TEXT NOT NULL,
            feature_kind TEXT NOT NULL,
            series_uid TEXT,
            series_label TEXT,
            modality TEXT,
            roi TEXT,
            features_json TEXT NOT NULL,
            metadata_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clinical_followup (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_name TEXT,
            case_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _feature_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    for key, value in row.items():
        if key in FEATURE_META_COLUMNS:
            continue
        safe_value = _json_safe(value)
        if safe_value is not None:
            payload[str(key)] = safe_value
    return payload


def save_feature_dataframe(
    df: pd.DataFrame,
    *,
    case_id: str,
    feature_kind: str,
    series_uid: str = "",
    series_label: str = "",
    modality: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """Save a feature matrix to the temporary SQLite database.

    Returns the number of ROI rows written.
    """
    if df is None or df.empty:
        return 0

    now = datetime.utcnow().isoformat(timespec="seconds")
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
    written = 0

    with get_connection(db_path) as conn:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            roi = str(row_dict.get("ROI", ""))
            row_series_uid = str(row_dict.get("SeriesUID", series_uid) or "")
            row_series_label = str(row_dict.get("Series", row_dict.get("SeriesLabel", series_label)) or "")
            row_modality = str(row_dict.get("Modality", modality) or "")
            row_feature_kind = str(row_dict.get("FeatureKind", feature_kind) or feature_kind)
            features = _feature_payload(row_dict)
            if not features:
                continue
            conn.execute(
                """
                INSERT INTO feature_results
                (created_at, case_id, feature_kind, series_uid, series_label,
                 modality, roi, features_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now, str(case_id), row_feature_kind, row_series_uid, row_series_label,
                    row_modality, roi, json.dumps(features, ensure_ascii=False),
                    metadata_json,
                ),
            )
            written += 1
        conn.commit()

    return written


def list_saved_results(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, case_id, feature_kind, series_label,
                   modality, roi, features_json
            FROM feature_results
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()

    records = []
    for row in rows:
        features = json.loads(row["features_json"])
        records.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "case_id": row["case_id"],
            "feature_kind": row["feature_kind"],
            "series_label": row["series_label"] or "",
            "modality": row["modality"] or "",
            "roi": row["roi"] or "",
            "feature_count": len(features),
        })
    return pd.DataFrame(records)


def load_feature_wide_table(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Return one modeling row per case with feature columns widened."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT case_id, feature_kind, series_label, roi, features_json
            FROM feature_results
            ORDER BY id
            """
        ).fetchall()

    by_case: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = row["case_id"]
        record = by_case.setdefault(case_id, {"case_id": case_id})
        feature_kind = row["feature_kind"] or "feature"
        series_label = row["series_label"] or "series"
        roi = row["roi"] or "ROI"
        features = json.loads(row["features_json"])
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                col = f"{feature_kind}|{series_label}|{roi}|{feature_name}"
                record[col] = value

    return pd.DataFrame(by_case.values())


def save_clinical_followup(
    df: pd.DataFrame,
    *,
    case_id_col: str,
    source_name: str = "clinical_followup",
    replace: bool = True,
    db_path: Optional[Path | str] = None,
) -> int:
    if df is None or df.empty or case_id_col not in df.columns:
        return 0

    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for _, row in df.iterrows():
        row_dict = {str(k): _json_safe(v) for k, v in row.to_dict().items()}
        case_id = row_dict.get(case_id_col)
        if case_id is None or str(case_id).strip() == "":
            continue
        rows.append((now, source_name, str(case_id), json.dumps(row_dict, ensure_ascii=False)))

    with get_connection(db_path) as conn:
        if replace:
            conn.execute("DELETE FROM clinical_followup")
        conn.executemany(
            """
            INSERT INTO clinical_followup
            (created_at, source_name, case_id, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def load_clinical_followup(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT case_id, payload_json
            FROM clinical_followup
            ORDER BY id
            """
        ).fetchall()

    records = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        payload["case_id"] = row["case_id"]
        records.append(payload)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.drop_duplicates(subset=["case_id"], keep="last")


def build_modeling_table(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    features = load_feature_wide_table(db_path)
    clinical = load_clinical_followup(db_path)
    if features.empty:
        return clinical
    if clinical.empty:
        return features
    return features.merge(clinical, on="case_id", how="left", suffixes=("", "_clinical"))


def clear_database(db_path: Optional[Path | str] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM feature_results")
        conn.execute("DELETE FROM clinical_followup")
        conn.commit()


def read_tabular_upload(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)
