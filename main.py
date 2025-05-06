"""
Main script for the Hubs project.

This script runs the data processing pipeline and performs analysis.
"""

import argparse
import logging
from pathlib import Path

from src.data.features import process_features_and_target
from src.data.transform import process_data
from src.models.train import train_and_evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    conversion_window_days: int = 28,
    skip_model: bool = False,
    model_output: str = None,
    validation_output: str = None,
    feature_importance_output: str = None,
    shap_output: str = None,
    shap_plots_dir: str = None,
    skip_shap: bool = False,
    threshold: float = 0.2,
):
    """
    Run the main data processing and analysis pipeline.

    Args:
        conversion_window_days: Number of days to look ahead for conversion
        skip_model: Whether to skip model training
        model_output: Custom path to save the model file
        validation_output: Custom path to save validation set with predictions
        feature_importance_output: Custom path to save feature importance data
        shap_output: Custom path to save SHAP values summary
        shap_plots_dir: Custom directory to save SHAP visualizations
        skip_shap: Whether to skip SHAP values calculation
        threshold: Classification threshold (lower values favor recall over precision)
    """
    logger.info("Starting data processing...")

    # Process the raw data
    data = process_data()

    # Create output directories if they don't exist
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)

    # Save the processed data
    processed_path = output_dir / "accounts_with_usage.parquet"
    data.to_parquet(processed_path)

    logger.info(f"Processed data saved to {processed_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Total accounts: {data['id'].nunique()}")

    # Create target variable
    logger.info(
        f"Creating target variable with {conversion_window_days}-day conversion window..."
    )
    data_with_target = process_features_and_target(
        processed_path, conversion_window_days
    )

    # Save the data with target
    target_path = (
        output_dir / f"data_with_target_{conversion_window_days}d.parquet"
    )
    data_with_target.to_parquet(target_path)

    logger.info(f"Data with target saved to {target_path}")
    logger.info(f"Final data shape: {data_with_target.shape}")

    # Show first few rows
    logger.info("\nSample data with target:")
    logger.info(f"\n{data_with_target.head()}")

    # Train model if not skipped
    if not skip_model:
        logger.info("Training XGBoost model...")
        # Use custom paths if provided, otherwise use default paths
        model_output_path = (
            Path(model_output)
            if model_output
            else models_dir / f"xgboost_model_{conversion_window_days}d.json"
        )
        validation_output_path = (
            Path(validation_output)
            if validation_output
            else output_dir
            / f"validation_with_predictions_{conversion_window_days}d.parquet"
        )
        feature_importance_path = (
            Path(feature_importance_output)
            if feature_importance_output
            else models_dir
            / f"feature_importance_{conversion_window_days}d.csv"
        )
        shap_output_path = (
            Path(shap_output)
            if shap_output
            else models_dir / f"shap_summary_{conversion_window_days}d.csv"
        )
        shap_plots_directory = (
            Path(shap_plots_dir)
            if shap_plots_dir
            else models_dir / f"shap_plots_{conversion_window_days}d"
        )

        # Train and evaluate model
        model, metrics = train_and_evaluate(
            data_path=target_path,
            model_output_path=model_output_path,
            validation_output_path=validation_output_path,
            feature_importance_path=feature_importance_path,
            shap_output_path=shap_output_path,
            shap_plots_dir=shap_plots_directory,
            test_size=0.2,
            threshold=threshold,
            calculate_shap=not skip_shap,
        )

        logger.info(
            f"Model training complete. Model saved to {model_output_path}"
        )
        logger.info(
            f"Validation set with predictions saved to {validation_output_path}"
        )
        logger.info(f"Feature importance saved to {feature_importance_path}")

        if not skip_shap:
            logger.info(f"SHAP values summary saved to {shap_output_path}")
            logger.info(f"SHAP visualizations saved to {shap_plots_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hubs data processing pipeline"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=28,
        help="Number of days to look ahead for conversion (default: 28)",
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip model training"
    )
    parser.add_argument(
        "--model-output", type=str, help="Custom path to save the model file"
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        help="Custom path to save validation set with predictions",
    )
    parser.add_argument(
        "--feature-importance-output",
        type=str,
        help="Custom path to save feature importance data",
    )
    parser.add_argument(
        "--shap-output",
        type=str,
        help="Custom path to save SHAP values summary",
    )
    parser.add_argument(
        "--shap-plots-dir",
        type=str,
        help="Custom directory to save SHAP visualizations",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP values calculation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Classification threshold (lower values favor recall, default: 0.2)",
    )

    args = parser.parse_args()

    main(
        conversion_window_days=args.window,
        skip_model=args.skip_model,
        model_output=args.model_output,
        validation_output=args.validation_output,
        feature_importance_output=args.feature_importance_output,
        shap_output=args.shap_output,
        shap_plots_dir=args.shap_plots_dir,
        skip_shap=args.skip_shap,
        threshold=args.threshold,
    )
