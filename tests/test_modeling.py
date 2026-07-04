import pandas as pd
import pytest

from src.modeling import infer_task_type, run_supervised_models


def test_infer_task_type():
    assert infer_task_type(pd.Series([1.2, 2.3, 3.4, 4.5, 5.6])) == "regression"
    assert infer_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"
    assert infer_task_type(pd.Series(["yes", "no", "yes"])) == "classification"


def test_run_supervised_models_regression_when_sklearn_available():
    pytest.importorskip("sklearn")

    data = pd.DataFrame({
        "case_id": [f"case-{i}" for i in range(20)],
        "feature_a": list(range(20)),
        "feature_b": [i % 3 for i in range(20)],
        "target": [float(i) * 1.5 for i in range(20)],
    })
    results = run_supervised_models(
        data,
        feature_cols=["feature_a", "feature_b"],
        target_col="target",
        task_type="regression",
        model_names=["Linear Regression"],
        validation_size=0.2,
        test_size=0.2,
    )

    assert results["split_sizes"] == {"train": 12, "validation": 4, "test": 4}
    assert results["metrics"]["Model"].iloc[0] == "Linear Regression"
    assert "Test R2" in results["metrics"].columns
