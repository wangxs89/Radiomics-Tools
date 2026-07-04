"""Machine-learning helpers for radiomics modeling workflows."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge",
    "LASSO",
    "ElasticNet",
    "Support Vector Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "K-Nearest Neighbors",
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "K-Nearest Neighbors",
]


def infer_task_type(y: pd.Series) -> str:
    """Infer supervised task type from the outcome column."""
    non_null = y.dropna()
    if non_null.empty:
        return "regression"
    if pd.api.types.is_numeric_dtype(non_null):
        unique_count = non_null.nunique()
        if unique_count <= min(10, max(2, len(non_null) // 5)):
            return "classification"
        return "regression"
    return "classification"


def build_preprocessor(
    X: pd.DataFrame,
    *,
    numeric_imputer: str = "median",
    scale_numeric: bool = True,
    encode_categorical: bool = True,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))

    if categorical_cols and encode_categorical:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical, categorical_cols))
    elif categorical_cols:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("cat", categorical, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Dict[str, pd.DataFrame | pd.Series]:
    from sklearn.model_selection import train_test_split

    if validation_size < 0 or test_size < 0 or validation_size + test_size >= 1:
        raise ValueError("Validation + test size must be less than 1.0")

    stratify_y = y if stratify and _can_stratify(y) else None
    temp_size = validation_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=stratify_y,
    )

    relative_test_size = test_size / temp_size
    temp_stratify_y = y_temp if stratify and _can_stratify(y_temp) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size,
        random_state=random_state, stratify=temp_stratify_y,
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts(dropna=False)
    return len(counts) > 1 and counts.min() >= 2


def make_supervised_model(name: str, task_type: str):
    if task_type == "classification":
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        mapping = {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
        return mapping[name]

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    mapping = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "LASSO": Lasso(max_iter=10000),
        "ElasticNet": ElasticNet(max_iter=10000),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    return mapping[name]


def run_supervised_models(
    data: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    task_type: str,
    model_names: Sequence[str],
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    numeric_imputer: str = "median",
    scale_numeric: bool = True,
) -> Dict[str, pd.DataFrame | Dict[str, int]]:
    from sklearn.pipeline import Pipeline

    work = data[list(feature_cols) + [target_col]].dropna(subset=[target_col]).copy()
    if len(work) < 5:
        raise ValueError("Need at least 5 rows with non-missing outcome values for modeling.")

    X = work[list(feature_cols)].copy()
    y = work[target_col].copy()
    if task_type == "classification":
        y = y.astype(str)

    splits = split_train_validation_test(
        X, y, validation_size=validation_size, test_size=test_size,
        random_state=random_state, stratify=(task_type == "classification"),
    )

    rows = []
    for model_name in model_names:
        estimator = make_supervised_model(model_name, task_type)
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(
                splits["X_train"],
                numeric_imputer=numeric_imputer,
                scale_numeric=scale_numeric,
            )),
            ("model", estimator),
        ])

        try:
            pipeline.fit(splits["X_train"], splits["y_train"])
            val_pred = pipeline.predict(splits["X_val"])
            test_pred = pipeline.predict(splits["X_test"])
            row = {"Model": model_name}
            row.update(_evaluate_predictions(
                splits["y_val"], val_pred, task_type, prefix="Validation", model=pipeline,
                X=splits["X_val"],
            ))
            row.update(_evaluate_predictions(
                splits["y_test"], test_pred, task_type, prefix="Test", model=pipeline,
                X=splits["X_test"],
            ))
            rows.append(row)
        except Exception as exc:
            rows.append({"Model": model_name, "Error": str(exc)})

    metrics = pd.DataFrame(rows)
    split_sizes = {
        "train": len(splits["X_train"]),
        "validation": len(splits["X_val"]),
        "test": len(splits["X_test"]),
    }
    return {"metrics": metrics, "split_sizes": split_sizes}


def _evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task_type: str,
    *,
    prefix: str,
    model=None,
    X: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    if task_type == "classification":
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

        result = {
            f"{prefix} Accuracy": accuracy_score(y_true, y_pred),
            f"{prefix} Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            f"{prefix} F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if model is not None and X is not None and len(pd.Series(y_true).unique()) == 2:
            try:
                y_score = model.predict_proba(X)[:, 1]
                result[f"{prefix} ROC AUC"] = roc_auc_score(y_true, y_score)
            except Exception:
                pass
        return result

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix} MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix} RMSE": float(np.sqrt(mse)),
        f"{prefix} R2": r2_score(y_true, y_pred),
    }


def run_kmeans_clustering(
    data: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    n_clusters: int = 3,
    numeric_imputer: str = "median",
    scale_numeric: bool = True,
) -> Dict[str, pd.DataFrame | Dict[str, float]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline

    if len(data) < n_clusters:
        raise ValueError("Number of rows must be at least the number of clusters.")

    X = data[list(feature_cols)].copy()
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(
            X, numeric_imputer=numeric_imputer, scale_numeric=scale_numeric,
        )),
        ("model", KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)),
    ])
    labels = pipeline.fit_predict(X)
    transformed = pipeline.named_steps["preprocess"].transform(X)
    metrics = {"Inertia": pipeline.named_steps["model"].inertia_}
    if n_clusters > 1 and len(set(labels)) > 1 and len(data) > n_clusters:
        metrics["Silhouette"] = silhouette_score(transformed, labels)
    assignments = pd.DataFrame({
        "case_id": data["case_id"].values if "case_id" in data.columns else data.index.astype(str),
        "cluster": labels,
    })
    return {"assignments": assignments, "metrics": metrics}


def candidate_feature_columns(data: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
    blocked = {"case_id", target_col}
    return [c for c in data.columns if c not in blocked]
