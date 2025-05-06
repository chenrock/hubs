"""
Model prediction module for the Hubs project.

This module provides functions to load a trained XGBoost model and make predictions.
"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Union[str, Path]) -> xgb.Booster:
    """
    Load a trained XGBoost model from a file.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded XGBoost model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))

    return model


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction by handling the same preprocessing
    steps that were used during training.

    Args:
        df: DataFrame with raw features

    Returns:
        DataFrame with features preprocessed for prediction
    """
    from src.data.features import transform_alexa_rank, transform_industry

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Apply ALEXA_RANK transformation
    result = transform_alexa_rank(result)

    # Apply INDUSTRY transformation
    result = transform_industry(result)

    # Handle any other feature transformations needed for prediction
    # For example, if you need to calculate percentage changes or cumulative features

    # If there are any preprocessing steps that involve creating
    # categorical dummies, they should be added here

    return result


def predict(
    model: xgb.Booster,
    data: pd.DataFrame,
    threshold: float = 0.2,
    keep_id: bool = True,
) -> pd.DataFrame:
    """
    Make predictions using a trained model.

    Args:
        model: Trained XGBoost model
        data: DataFrame with features (excluding target)
        threshold: Classification threshold for binary predictions
        keep_id: Whether to keep the ID column in the results

    Returns:
        DataFrame with original data and added prediction columns
    """
    # Make a copy to avoid modifying the original
    result = data.copy()

    # Store ID column if present and requested
    id_col = None
    if "id" in result.columns and keep_id:
        id_col = result["id"].copy()

    # Prepare features for DMatrix
    X = (
        result.drop(columns=["id"])
        if "id" in result.columns
        else result.copy()
    )

    # Convert to DMatrix for prediction
    dmatrix = xgb.DMatrix(X)

    # Make predictions
    result["predicted_proba"] = model.predict(dmatrix)
    result["predicted_class"] = (
        result["predicted_proba"] >= threshold
    ).astype(int)

    # Log prediction summary
    positive_pct = result["predicted_class"].mean() * 100
    logger.info(f"Made predictions for {len(result)} rows")
    logger.info(f"Predicted positive rate: {positive_pct:.2f}%")

    # If we're not keeping all columns, create a smaller result DataFrame
    if not keep_id:
        if id_col is not None:
            return pd.DataFrame(
                {
                    "id": id_col,
                    "predicted_proba": result["predicted_proba"],
                    "predicted_class": result["predicted_class"],
                }
            )
        else:
            return pd.DataFrame(
                {
                    "predicted_proba": result["predicted_proba"],
                    "predicted_class": result["predicted_class"],
                }
            )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hubs model prediction")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved model file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data for prediction"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save predictions"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Classification threshold"
    )

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Load the data
    data = pd.read_parquet(args.data)

    # Make predictions
    predictions = predict(model, data, threshold=args.threshold)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    predictions.to_parquet(output_path)
    logger.info(f"Predictions saved to {output_path}")
