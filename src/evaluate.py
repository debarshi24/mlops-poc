# src/evaluate.py
import joblib
import pandas as pd
import logging
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO)


def _get_model_feature_names(model):
    """
    Try to find the feature names the model was trained with.
    Returns a list or None.
    """
    # Common locations depending on whether model is a Pipeline or estimator
    # 1) If model is a Pipeline, try final estimator
    try:
        if hasattr(model, "named_steps"):
            # get final estimator
            final = list(model.named_steps.values())[-1]
        else:
            final = model
    except Exception:
        final = model

    # Try attribute feature_names_in_ (scikit-learn >= 1.0)
    if hasattr(final, "feature_names_in_"):
        try:
            return list(final.feature_names_in_)
        except Exception:
            pass

    # Some sklearn wrappers store feature names in named_steps['preprocessor'] or scaler
    try:
        # attempt to read scaler.feature_names_in_ if exists
        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            scaler = model.named_steps["scaler"]
            if hasattr(scaler, "feature_names_in_"):
                return list(scaler.feature_names_in_)
    except Exception:
        pass

    return None


def _align_test_features(X_test: pd.DataFrame, model_feature_names):
    """
    Align X_test (one-hot encoded) to the model_feature_names:
    - add missing columns with zeros
    - drop extra columns not in model_feature_names
    - reorder columns to match model_feature_names
    """
    if model_feature_names is None:
        return X_test

    # Ensure columns present
    missing_cols = [c for c in model_feature_names if c not in X_test.columns]
    for c in missing_cols:
        X_test[c] = 0

    # Drop extras
    extra_cols = [c for c in X_test.columns if c not in model_feature_names]
    if extra_cols:
        X_test = X_test.drop(columns=extra_cols)

    # Reorder
    X_test = X_test.loc[:, model_feature_names]

    return X_test


def evaluate_model(model_path: str, test_csv_path: str) -> Dict[str, float]:
    """
    Load the trained model and test CSV and compute metrics.

    Args:
        model_path: local path to model.joblib (joblib dump of a sklearn Pipeline)
        test_csv_path: local path to test.csv

    Returns:
        metrics dict: {'accuracy':..., 'f1':..., 'precision':..., 'recall':...}
    """
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading test data from %s", test_csv_path)
    df = pd.read_csv(test_csv_path)

    # Determine target column
    if "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]

    y_true = df[target_col]
    X_test = df.drop(columns=[target_col])

    # Convert categorical via get_dummies (same approach used in train.py)
    X_test = pd.get_dummies(X_test)

    # Try prediction directly first (best-case)
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.warning("Direct prediction failed: %s. Attempting feature alignment...", e)
        # Try to align test features to model's expected features
        model_feature_names = _get_model_feature_names(model)
        if model_feature_names is not None:
            logger.info("Model feature names found, aligning test features.")
            X_test = _align_test_features(X_test, model_feature_names)
            try:
                y_pred = model.predict(X_test)
            except Exception as e2:
                logger.exception("Prediction failed after alignment: %s", e2)
                raise
        else:
            # No feature names discovered; try a safer fallback: align columns by intersection
            shared_cols = [c for c in X_test.columns if hasattr(model, "coef_") and c in getattr(model, "coef_", {})]
            # If fallback can't find anything meaningful, attempt prediction again and let it error
            try:
                y_pred = model.predict(X_test)
            except Exception:
                logger.exception("Unable to predict with given test features and no model feature names found.")
                raise

    # Compute metrics
    # Decide binary vs multiclass for f1/precision/recall averaging
    unique_labels = set(y_true)
    if len(unique_labels) == 2:
        average = "binary"
    else:
        average = "macro"

    # Ensure arrays (handle pandas Series)
    y_true_list = y_true.values
    y_pred_list = y_pred

    accuracy = float(accuracy_score(y_true_list, y_pred_list))
    f1 = float(f1_score(y_true_list, y_pred_list, average=average))
    precision = float(precision_score(y_true_list, y_pred_list, average=average, zero_division=0))
    recall = float(recall_score(y_true_list, y_pred_list, average=average, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    logger.info("Evaluation metrics computed: %s", metrics)
    return metrics
