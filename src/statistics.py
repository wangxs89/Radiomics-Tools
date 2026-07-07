"""Statistical analysis module for radiomics features.

Provides ICC (Intraclass Correlation Coefficient), LASSO feature selection,
and correlation analysis for radiomics studies.
"""
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats


def compute_icc(
    data: pd.DataFrame,
    icc_type: str = "ICC(2,1)",
    unit: str = "single",
) -> pd.DataFrame:
    """Compute Intraclass Correlation Coefficient (ICC) for feature reliability.

    ICC assesses the reliability of radiomics features across repeated measurements
    (e.g., test-retest, inter-observer, intra-observer variability).

    Args:
        data: DataFrame with features. Should have repeated measurements as rows
              and features as columns. For test-retest, each subject should have
              2+ rows.
        icc_type: ICC model type:
            - "ICC(1,1)": One-way random, single measures
            - "ICC(1,k)": One-way random, average measures
            - "ICC(2,1)": Two-way random, single measures (default)
            - "ICC(2,k)": Two-way random, average measures
            - "ICC(3,1)": Two-way mixed, single measures
            - "ICC(3,k)": Two-way mixed, average measures
        unit: "single" or "average"

    Returns:
        DataFrame with ICC values, F-statistics, and p-values for each feature.
    """
    # Determine k (number of ratings per subject)
    n_subjects = len(data) // 2  # Assuming 2 measurements per subject
    k = 2

    results = []

    for col in data.select_dtypes(include=[np.number]).columns:
        values = data[col].values

        # Reshape into (subjects, ratings)
        if len(values) % k != 0:
            results.append({
                'Feature': col,
                'ICC': np.nan,
                'F_stat': np.nan,
                'p_value': np.nan,
                'CI_lower': np.nan,
                'CI_upper': np.nan,
            })
            continue

        ratings = values.reshape(n_subjects, k)

        # Compute ICC
        try:
            icc_val, f_stat, p_val, ci_lower, ci_upper = _compute_single_icc(
                ratings, icc_type, unit
            )
            results.append({
                'Feature': col,
                'ICC': icc_val,
                'F_stat': f_stat,
                'p_value': p_val,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
            })
        except Exception:
            results.append({
                'Feature': col,
                'ICC': np.nan,
                'F_stat': np.nan,
                'p_value': np.nan,
                'CI_lower': np.nan,
                'CI_upper': np.nan,
            })

    return pd.DataFrame(results)


def _compute_single_icc(
    ratings: np.ndarray,
    icc_type: str = "ICC(2,1)",
    unit: str = "single",
) -> Tuple[float, float, float, float, float]:
    """Compute ICC for a single feature.

    Returns:
        (icc, f_stat, p_value, ci_lower, ci_upper)
    """
    n, k = ratings.shape

    # Subject means
    subject_means = ratings.mean(axis=1)
    # Overall mean
    grand_mean = ratings.mean()

    # Sum of squares
    SS_subject = k * np.sum((subject_means - grand_mean) ** 2)
    SS_error = np.sum((ratings - subject_means[:, np.newaxis]) ** 2)
    SS_total = np.sum((ratings - grand_mean) ** 2)

    # Degrees of freedom
    df_subject = n - 1
    df_error = n * (k - 1)

    # Mean squares
    MS_subject = SS_subject / df_subject
    MS_error = SS_error / df_error

    # Compute ICC based on type
    if icc_type in ["ICC(1,1)", "ICC(1,k)"]:
        # One-way random
        MS_between = MS_subject
        icc = (MS_between - MS_error) / (MS_between + (k - 1) * MS_error)
    elif icc_type in ["ICC(2,1)", "ICC(2,k)"]:
        # Two-way random
        MS_rater = np.sum((ratings.mean(axis=0) - grand_mean) ** 2) / (k - 1)
        icc = (MS_subject - MS_error) / (MS_subject + (k - 1) * MS_error +
                                         k * (MS_rater - MS_error) / n)
    elif icc_type in ["ICC(3,1)", "ICC(3,k)"]:
        # Two-way mixed
        icc = (MS_subject - MS_error) / (MS_subject + (k - 1) * MS_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")

    # F-statistic
    f_stat = MS_subject / MS_error if MS_error > 0 else np.nan
    p_value = 1 - stats.f.cdf(f_stat, df_subject, df_error) if not np.isnan(f_stat) else np.nan

    # Confidence interval (simplified)
    ci_lower = max(0, icc - 1.96 * np.sqrt(2 * (1 - icc) ** 2 / (n - 1))) if not np.isnan(icc) else np.nan
    ci_upper = min(1, icc + 1.96 * np.sqrt(2 * (1 - icc) ** 2 / (n - 1))) if not np.isnan(icc) else np.nan

    if unit == "average":
        icc = (k * icc) / (1 + (k - 1) * icc) if not np.isnan(icc) else np.nan

    return icc, f_stat, p_value, ci_lower, ci_upper


def lasso_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: Optional[float] = None,
    cv_folds: int = 5,
    max_iter: int = 10000,
) -> dict:
    """LASSO feature selection for radiomics features.

    Args:
        X: Feature matrix (samples × features)
        y: Target variable (binary outcome or continuous)
        alpha: Regularization strength. If None, determined by cross-validation.
        cv_folds: Number of cross-validation folds
        max_iter: Maximum iterations for LASSO solver

    Returns:
        dict with keys:
            - 'selected_features': List of selected feature names
            - 'coefficients': DataFrame with feature coefficients
            - 'alpha': Alpha value used
            - 'scores': Cross-validation scores (if alpha is None)
    """
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler

    # Handle missing values
    X_clean = X.dropna(axis=1)
    valid_cols = X_clean.columns.tolist()

    if len(valid_cols) == 0:
        return {
            'selected_features': [],
            'coefficients': pd.DataFrame(),
            'alpha': alpha,
            'scores': [],
        }

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean.values)

    # Align y with X
    y_aligned = y.loc[X_clean.index].values

    if alpha is None:
        # Use cross-validation to find optimal alpha
        model = LassoCV(
            cv=cv_folds,
            max_iter=max_iter,
            random_state=42,
        )
        model.fit(X_scaled, y_aligned)
        alpha_opt = model.alpha_
        scores = model.mse_path_.mean(axis=1)
    else:
        alpha_opt = alpha
        scores = []
        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        model.fit(X_scaled, y_aligned)

    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': valid_cols,
        'Coefficient': model.coef_,
    })
    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)

    # Select features with non-zero coefficients
    selected = coefficients[coefficients['Coefficient'] != 0]['Feature'].tolist()

    return {
        'selected_features': selected,
        'coefficients': coefficients,
        'alpha': alpha_opt,
        'scores': scores,
    }


def correlation_analysis(
    data: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.9,
) -> dict:
    """Perform correlation analysis on radiomics features.

    Args:
        data: Feature matrix (samples × features)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Correlation threshold for identifying highly correlated pairs

    Returns:
        dict with keys:
            - 'correlation_matrix': Full correlation matrix
            - 'high_correlations': Pairs with correlation > threshold
            - 'feature_clusters': Clusters of highly correlated features
    """
    # Compute correlation matrix
    corr_matrix = data.select_dtypes(include=[np.number]).corr(method=method)

    # Find highly correlated pairs
    high_corr_pairs = []
    features = corr_matrix.columns.tolist()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'Feature1': features[i],
                    'Feature2': features[j],
                    'Correlation': corr_val,
                })

    high_corr_df = pd.DataFrame(high_corr_pairs)
    if len(high_corr_df) > 0:
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)

    # Simple clustering: group features by high correlation
    clusters = []
    used = set()

    for _, row in high_corr_df.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']
        if f1 not in used and f2 not in used:
            clusters.append([f1, f2])
            used.add(f1)
            used.add(f2)
        elif f1 not in used:
            for cluster in clusters:
                if f2 in cluster:
                    cluster.append(f1)
                    used.add(f1)
                    break
        elif f2 not in used:
            for cluster in clusters:
                if f1 in cluster:
                    cluster.append(f2)
                    used.add(f2)
                    break

    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_df,
        'feature_clusters': clusters,
    }


def icc_reliability_classification(icc_value: float) -> str:
    """Classify ICC value according to reliability guidelines.

    Args:
        icc_value: ICC value (0-1)

    Returns:
        Reliability classification string
    """
    if icc_value < 0:
        return "Poor (negative)"
    elif icc_value < 0.5:
        return "Poor"
    elif icc_value < 0.75:
        return "Moderate"
    elif icc_value < 0.9:
        return "Good"
    else:
        return "Excellent"
