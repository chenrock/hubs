"""
Model training module for the Hubs project.

This module provides functions to train an XGBoost classification model
on the prepared data with target variable.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data_with_target(
    data_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load the data with target variable.

    Args:
        data_path: Path to the data file with target

    Returns:
        DataFrame containing data with target
    """
    if data_path is None:
        data_path = Path(
            ".", "data", "processed", "data_with_target_28d.parquet"
        )
    else:
        data_path = Path(data_path)

    logger.info(f"Loading data with target from {data_path}")
    data = pd.read_parquet(data_path)

    return data


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training.

    Args:
        df: DataFrame with data and target

    Returns:
        DataFrame with prepared features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Drop future data (we don't want to use future info in our model)
    result = result.drop(columns=["CLOSEDATE", "MRR"])

    # Convert categorical features to one-hot encoding
    categorical_cols = ["EMPLOYEE_RANGE", "industry_bucket"]
    for col in categorical_cols:
        if col in result.columns:
            # Create one-hot encoding with dummy_na=True to create a column for NaN values
            dummies = pd.get_dummies(result[col], prefix=col, dummy_na=True)

            # Log missing value information
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                logger.info(
                    f"Column {col}: {missing_count} missing values ({missing_count / len(result):.2%})"
                )
                logger.info(
                    f"Created separate indicator column {col}_nan for missing values"
                )

            # Add to result and drop original column
            result = pd.concat([result, dummies], axis=1)
            result = result.drop(columns=[col])

    # Drop original INDUSTRY column since we're using the bucketed version
    if "INDUSTRY" in result.columns:
        result = result.drop(columns=["INDUSTRY"])

    # Handle ALEXA_RANK separately since we've created the transformed version
    if "ALEXA_RANK" in result.columns and "alexa_rank_score" in result.columns:
        # Now that we have alexa_rank_score, we can drop the original column
        result = result.drop(columns=["ALEXA_RANK"])
        logger.info(
            "Using min-max scaled alexa_rank_score feature instead of raw ALEXA_RANK"
        )

    # WHEN_TIMESTAMP to useful features (month, year)
    result["month"] = result["WHEN_TIMESTAMP"].dt.month
    result["year"] = result["WHEN_TIMESTAMP"].dt.year

    # Drop columns not used in training
    cols_to_drop = ["WHEN_TIMESTAMP"]
    result = result.drop(columns=cols_to_drop)

    logger.info(f"Prepared features with shape: {result.shape}")
    logger.info(f"Feature columns: {result.columns.tolist()}")

    return result


def split_data_by_id(
    df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets by account ID.

    Args:
        df: DataFrame with features and target
        id_col: Name of the ID column
        target_col: Name of the target column
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Ensure we have the ID column
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame")

    # Create a GroupShuffleSplit which ensures that all rows with the same ID
    # will be in either train or test, but not both
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    # Get the split indices
    train_indices, val_indices = next(splitter.split(df, groups=df[id_col]))

    # Create train and validation sets
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Verify that IDs don't overlap
    train_ids = set(train_df[id_col].unique())
    val_ids = set(val_df[id_col].unique())
    overlap = train_ids.intersection(val_ids)

    if overlap:
        logger.warning(
            f"Found {len(overlap)} IDs in both train and validation sets: {overlap}"
        )
    else:
        logger.info("No overlap in IDs between train and validation sets")

    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # Log split information
    train_ids = X_train[id_col].nunique()
    val_ids = X_val[id_col].nunique()

    logger.info(f"Training set: {len(X_train)} rows, {train_ids} unique IDs")
    logger.info(f"Validation set: {len(X_val)} rows, {val_ids} unique IDs")
    logger.info(f"Training set positive rate: {y_train.mean():.2%}")
    logger.info(f"Validation set positive rate: {y_val.mean():.2%}")

    return X_train, X_val, y_train, y_val


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
) -> xgb.Booster:
    """
    Train an XGBoost classification model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: XGBoost parameters (optional)

    Returns:
        Trained XGBoost model
    """
    # Remove ID column for training if present
    X_train_model = (
        X_train.drop(columns=["id"])
        if "id" in X_train.columns
        else X_train.copy()
    )
    X_val_model = (
        X_val.drop(columns=["id"]) if "id" in X_val.columns else X_val.copy()
    )

    # Default parameters if none provided
    if params is None:
        # Calculate scale_pos_weight - higher values favor recall over precision
        # Use an even higher multiplier (5.0) to heavily favor recall
        pos_scale = 5.0 * (1 - y_train.mean()) / y_train.mean()

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "error"],
            "eta": 0.05,  # Smaller learning rate for better convergence
            "max_depth": 8,  # Deeper trees to capture complex patterns
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,  # Keep this low to allow more splits
            "scale_pos_weight": pos_scale,  # Very high focus on positive cases
            "gamma": 0.1,  # Minimal pruning
            "seed": 42,
        }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_model, label=y_train)
    dval = xgb.DMatrix(X_val_model, label=y_val)

    # Specify evaluation sets
    evals = [(dtrain, "train"), (dval, "validation")]

    # Train the model
    logger.info("Training XGBoost model with recall-focused settings...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1500,  # More rounds for better convergence with lower learning rate
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    logger.info(f"Best iteration: {model.best_iteration}")

    return model


def evaluate_model(
    model: xgb.Booster, X: pd.DataFrame, y: pd.Series, threshold: float = 0.2
) -> Dict:
    """
    Evaluate the trained model.

    Args:
        model: Trained XGBoost model
        X: Features to evaluate
        y: True labels
        threshold: Classification threshold (lower values favor recall)

    Returns:
        Dictionary of evaluation metrics
    """
    # Remove ID column for prediction if present
    X_eval = X.drop(columns=["id"]) if "id" in X.columns else X.copy()

    # Convert to DMatrix for prediction
    dtest = xgb.DMatrix(X_eval)

    # Get predictions (probabilities)
    y_pred_proba = model.predict(dtest)

    # Convert to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate standard metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "pr_auc": average_precision_score(
            y, y_pred_proba
        ),  # Precision-Recall AUC
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    # Log detailed classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y, y_pred))

    # Log metrics
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            logger.info(f"{metric}: {value:.4f}")

    logger.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")

    # Log the threshold used
    logger.info(f"Classification threshold: {threshold}")

    # Calculate precision-recall at different thresholds
    precision_curve, recall_curve, thresholds = precision_recall_curve(
        y, y_pred_proba
    )

    # Simply report recall at various thresholds for reference
    logger.info("\nRecall at different thresholds:")
    test_thresholds = [0.5, 0.3, 0.2, 0.1, 0.05]
    for test_t in test_thresholds:
        test_preds = (y_pred_proba >= test_t).astype(int)
        test_recall = recall_score(y, test_preds)
        test_precision = precision_score(y, test_preds)
        logger.info(
            f"  Threshold {test_t:.2f}: Recall = {test_recall:.4f}, Precision = {test_precision:.4f}"
        )

    return metrics


def get_feature_importance(
    model: xgb.Booster, feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from the trained model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    # Get feature importance scores
    importance = model.get_score(importance_type="gain")

    # Convert to DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": list(importance.keys()),
            "importance": list(importance.values()),
        }
    )

    # Sort by importance
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    # Add relative importance
    importance_df["relative_importance"] = (
        importance_df["importance"] / importance_df["importance"].sum()
    )

    logger.info("\nTop 10 important features:")
    logger.info(importance_df.head(10))

    return importance_df


def calculate_shap_values(
    model: xgb.Booster,
    X: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    save_plots: bool = True,
    plot_top_n: int = 20,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate SHAP (SHapley Additive exPlanations) values for the model.

    Args:
        model: Trained XGBoost model
        X: Features to explain (without ID column)
        output_dir: Directory to save SHAP plots (if save_plots is True)
        save_plots: Whether to save SHAP plots to files
        plot_top_n: Number of top features to include in plots

    Returns:
        Tuple containing:
        - SHAP values as numpy array
        - DataFrame with SHAP values summary
    """
    # Remove ID column if present
    X_shap = X.drop(columns=["id"]) if "id" in X.columns else X.copy()

    # Create a tree explainer
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for all instances
    shap_values = explainer.shap_values(X_shap)

    # Create summary DataFrame
    shap_summary = pd.DataFrame(
        {
            "feature": X_shap.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )

    # Sort by mean absolute SHAP value
    shap_summary = shap_summary.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    # Log top features by SHAP values
    logger.info("\nTop 10 features by mean absolute SHAP value:")
    logger.info(shap_summary.head(10))

    # Create and save plots if requested
    if save_plots and output_dir is not None:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Create summary plot (bar chart of mean SHAP values)
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values,
                    X_shap,
                    plot_type="bar",
                    max_display=plot_top_n,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(
                    output_dir / "shap_summary_bar.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to create SHAP bar plot: {str(e)}")
                plt.close()

            # Create summary plot (beeswarm plot)
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values, X_shap, max_display=plot_top_n, show=False
                )
                plt.tight_layout()
                plt.savefig(
                    output_dir / "shap_summary_beeswarm.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as e:
                logger.warning(
                    f"Failed to create SHAP beeswarm plot: {str(e)}"
                )
                plt.close()

            # Create dependence plots for top 5 features
            for i, feature in enumerate(shap_summary["feature"].head(5)):
                try:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        feature, shap_values, X_shap, show=False
                    )
                    plt.tight_layout()
                    plt.savefig(
                        output_dir / f"shap_dependence_{feature}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
                except Exception as e:
                    logger.warning(
                        f"Failed to create dependence plot for {feature}: {str(e)}"
                    )
                    plt.close()

            logger.info(f"SHAP plots saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create SHAP visualization: {str(e)}")
            # Continue execution even if visualization fails

    # Create summary DataFrame with normalized values
    shap_summary["normalized_mean_abs_shap"] = (
        shap_summary["mean_abs_shap"] / shap_summary["mean_abs_shap"].sum()
    )

    return shap_values, shap_summary


def train_and_evaluate(
    data_path: Optional[str | Path] = None,
    model_output_path: Optional[str | Path] = None,
    validation_output_path: Optional[str | Path] = None,
    feature_importance_path: Optional[str | Path] = None,
    shap_output_path: Optional[str | Path] = None,
    shap_plots_dir: Optional[str | Path] = None,
    test_size: float = 0.2,
    threshold: float = 0.3,  # Lower threshold to favor recall over precision
    xgb_params: Optional[Dict] = None,
    calculate_shap: bool = True,
) -> Tuple[xgb.Booster, Dict]:
    """
    Train and evaluate an XGBoost model on the prepared data.

    Args:
        data_path: Path to the data file with target
        model_output_path: Path to save the trained model
        validation_output_path: Path to save validation set with predictions
        feature_importance_path: Path to save feature importance
        shap_output_path: Path to save SHAP values summary
        shap_plots_dir: Directory to save SHAP visualizations
        test_size: Proportion of data to use for validation
        threshold: Classification threshold
        xgb_params: XGBoost parameters (optional)
        calculate_shap: Whether to calculate SHAP values

    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Load data with target
    data = load_data_with_target(data_path)

    # Prepare features
    data_features = prepare_features(data)

    # Split data by account ID
    X_train, X_val, y_train, y_val = split_data_by_id(
        data_features, test_size=test_size
    )

    # Train model
    model = train_xgboost_model(
        X_train, y_train, X_val, y_val, params=xgb_params
    )

    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val, threshold=threshold)

    # Get feature importance
    feature_names = [col for col in X_train.columns if col != "id"]
    feature_importance = get_feature_importance(model, feature_names)

    # Calculate SHAP values if requested
    if calculate_shap:
        try:
            # Determine SHAP plots directory
            if shap_plots_dir is None and model_output_path is not None:
                # Default to a folder next to the model file
                shap_plots_dir = Path(model_output_path).parent / "shap_plots"

            # Calculate SHAP values
            _, shap_summary = calculate_shap_values(
                model,
                X_val,
                output_dir=shap_plots_dir,
                save_plots=shap_plots_dir is not None,
            )

            # Save SHAP summary to file if path provided
            if shap_output_path is not None:
                try:
                    shap_output_path = Path(shap_output_path)
                    shap_output_path.parent.mkdir(exist_ok=True, parents=True)
                    shap_summary.to_csv(shap_output_path, index=False)
                    logger.info(
                        f"SHAP values summary saved to {shap_output_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save SHAP summary to file: {str(e)}"
                    )
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {str(e)}")
            logger.info("Continuing without SHAP analysis")

    # Save model if output path is provided
    if model_output_path is not None:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(exist_ok=True, parents=True)
        model.save_model(str(model_output_path))
        logger.info(f"Model saved to {model_output_path}")

    # Save validation set with predictions
    if validation_output_path is not None:
        # Make a copy of validation set with ID column
        val_with_preds = X_val.copy()

        # Add true labels
        val_with_preds["actual_target"] = y_val.values

        # Get predictions (probabilities)
        X_val_pred = (
            X_val.drop(columns=["id"])
            if "id" in X_val.columns
            else X_val.copy()
        )
        dval = xgb.DMatrix(X_val_pred)
        val_with_preds["predicted_proba"] = model.predict(dval)

        # Add binary predictions using threshold
        val_with_preds["predicted_class"] = (
            val_with_preds["predicted_proba"] >= threshold
        ).astype(int)

        # Save to file
        validation_output_path = Path(validation_output_path)
        validation_output_path.parent.mkdir(exist_ok=True, parents=True)
        val_with_preds.to_parquet(validation_output_path)
        logger.info(
            f"Validation set with predictions saved to {validation_output_path}"
        )

    # Save feature importance
    if feature_importance_path is not None:
        feature_importance_path = Path(feature_importance_path)
        feature_importance_path.parent.mkdir(exist_ok=True, parents=True)
        feature_importance.to_csv(feature_importance_path, index=False)
        logger.info(f"Feature importance saved to {feature_importance_path}")

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hubs modeling pipeline")
    parser.add_argument("--data", type=str, help="Path to data with target")
    parser.add_argument(
        "--output", type=str, help="Path to save trained model"
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        help="Path to save validation set with predictions",
    )
    parser.add_argument(
        "--feature-importance-output",
        type=str,
        help="Path to save feature importance data",
    )
    parser.add_argument(
        "--shap-output",
        type=str,
        help="Path to save SHAP values summary",
    )
    parser.add_argument(
        "--shap-plots-dir",
        type=str,
        help="Directory to save SHAP visualizations",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP values calculation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size for validation split",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )

    args = parser.parse_args()

    data_path = args.data
    model_output_path = args.output or Path(
        ".", "models", "xgboost_model.json"
    )

    # Train and evaluate model
    train_and_evaluate(
        data_path=data_path,
        model_output_path=model_output_path,
        validation_output_path=args.validation_output,
        feature_importance_path=args.feature_importance_output,
        shap_output_path=args.shap_output,
        shap_plots_dir=args.shap_plots_dir,
        test_size=args.test_size,
        threshold=args.threshold,
        calculate_shap=not args.skip_shap,
    )
