"""
Feature engineering module for the Hubs project.

This module provides functions to create features and target variables from the joined data.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BUCKET_MAP = {
    # case-insensitive
    "Software & IT": [
        "computer_software",
        "technology - software",
        "software / it",
        "information_technology_and_services",
        "technology",
        "saas",
        "information technology & services",
        "telecommunications",
        "technology - hardware and equipment",
        "computer_networking",
        "computer_hardware",
        "computer_network_security",
        "technology - hardware & equipment",
        "internet",
        "telecom & mobile communications",
        "software",
        "computers, networking, software, technol",
    ],
    "Marketing & Advertising": [
        "marketing_and_advertising",
        "marketing services",
        "advertising",
        "advertising & marketing services",
        "mktg / pr / web / seo",
        "asesoría de marketing",
        "public relations and communications",
        "advertising and marketing",
    ],
    "Education & Non‑profit": [
        "higher_education",
        "education_management",
        "primary_secondary_education",
        "e_learning",
        "non-profit/educational institution",
        "non profit/edu/.gov",
        "civic_social_organization",
        "religious institutions",
        "non_profit_organization_management",
        "education",
        "non-profit",
        "non profit",
        "libraries",
    ],
    "Financial Services": [
        "financial_services",
        "banking",
        "banking/financial services",
        "finance & insurance",
        "insurance",
        "capital_markets",
        "investment_management",
        "finance and insurance",
        "banking and finance",
        "finance",
    ],
    "Real Estate & Construction": [
        "real_estate",
        "construction",
        "building, construction and architecture",
        "architecture_planning",
        "architecture",
        "commercial_real_estate",
        "building_materials",
        "real estate & construction",
        "real estate",
        "civil_engineering",
    ],
    "Consumer Goods & Retail": [
        "retail",
        "consumer_services",
        "consumer products",
        "consumer_goods",
        "consumer_electronics",
        "apparel_fashion",
        "cosmetics",
        "furniture",
        "food_beverages",
        "ecommerce",
        "e- commerce",
        "comercio electrónico",
        "wine_and_spirits",
        "food_production",
        "sporting_goods",
        "tiendas",
        "wholesale",
    ],
    "Professional & Consulting Services": [
        "consulting/advisory",
        "management_consulting",
        "business consulting",
        "professional_training_coaching",
        "legal_services",
        "accounting",
        "staffing_and_recruiting",
        "human_resources",
        "public_relations_and_communications",
        "law_practice",
        "recruiting/staffing",
        "business services - general",
        "coaching/business services",
        "consulting",
        "translation_and_localization",
        "market_research",
        "security_and_investigations",
        "design",
    ],
    "Healthcare & Wellness": [
        "hospital_health_care",
        "healthcare",
        "medical_devices",
        "medical_practice",
        "health_wellness_and_fitness",
        "mental_health_care",
        "biotechnology",
        "health/fitness",
        "healthcare - services/providers",
        "pharmaceuticals",
        "heath care (products or services)",
    ],
    "Hospitality & Travel": [
        "leisure_travel_tourism",
        "hospitality",
        "restaurants",
        "airlines_aviation",
        "travel",
        "reise",
        "events_services",
        "recreational_facilities_and_services",
        "gambling_casinos",
    ],
    "Manufacturing & Industrial": [
        "manufacturing",
        "machinery",
        "chemicals",
        "mechanical_or_industrial_engineering",
        "electrical_electronic_manufacturing",
        "semiconductors",
        "industrial_automation",
        "automotive",
        "oil_energy",
        "printing",
        "manufacturing & wholesale",
        "textiles",
        "packaging_and_containers",
        "glass_ceramics_concrete",
    ],
    "Media & Entertainment": [
        "entertainment",
        "media_production",
        "broadcast_media",
        "newspapers",
        "music",
        "photography",
        "online_media",
        "media & publishing",
        "publishing",
        "media",
        "sports",
    ],
    "Transportation & Logistics": [
        "transportation_trucking_railroad",
        "logistics_and_supply_chain",
        "maritime",
        "aviation_aerospace",
        "automotive & transport",
        "import_and_export",
    ],
    "Agriculture & Environment": [
        "renewables_environment",
        "environmental_services",
        "farming",
        "agriculture",
    ],
    "Government & Public Sector": [
        "government_administration",
        "law_enforcement",
        "judiciary",
        "defense_space",
        "government, public admin",
    ],
    "Research & Development": [
        "research",
        "program_development",
        "international_trade_and_development",
        "information_services",
    ],
    "Other": [
        "ustedes",
        "individual_family_services",
        "b2c services",
        "outsourcing_offshoring",
        "other services",
        "facilities_services",
        "cannabinoid",
        "utilities",
    ],
}


def load_processed_data(
    data_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load the processed data from a parquet file.

    Args:
        data_path: Path to the processed data file

    Returns:
        DataFrame containing processed data
    """
    if data_path is None:
        data_path = Path(
            ".", "data", "processed", "accounts_with_usage.parquet"
        )
    else:
        data_path = Path(data_path)

    logger.info(f"Loading processed data from {data_path}")
    data = pd.read_parquet(data_path)

    return data


def prepare_data_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for feature engineering by sorting appropriately.

    Args:
        df: DataFrame with joined account and usage data

    Returns:
        DataFrame sorted by id and WHEN_TIMESTAMP
    """
    # Sort by id and WHEN_TIMESTAMP
    sorted_data = df.sort_values(by=["id", "WHEN_TIMESTAMP"])

    return sorted_data


def create_target_variable(
    df: pd.DataFrame, window_days: int = 28
) -> pd.DataFrame:
    """
    Create the target variable based on CLOSEDATE and WHEN_TIMESTAMP.

    The target is 1 if:
    - There is a CLOSEDATE (customer)
    - The CLOSEDATE is within the specified window_days after the WHEN_TIMESTAMP

    The target is 0 if:
    - CLOSEDATE is null (non-customer)
    - CLOSEDATE is outside the window_days after WHEN_TIMESTAMP

    For each ID, all rows after the first row where target = 1 are dropped.
    If an ID never has target = 1, all rows for that ID are kept.

    Args:
        df: DataFrame with joined account and usage data (sorted by id and WHEN_TIMESTAMP)
        window_days: Number of days to look ahead for conversion

    Returns:
        DataFrame with added target variable and rows after conversion dropped
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Create a timedelta object for the window
    window = timedelta(days=window_days)

    # Create the target variable
    # 1. If CLOSEDATE is null, target is 0
    # 2. If WHEN_TIMESTAMP + window >= CLOSEDATE, target is 1 (conversion within window)
    # 3. Otherwise, target is 0 (conversion outside window or no conversion)
    result["target"] = 0  # Default to 0

    # For rows where CLOSEDATE is not null, check if it's within the window
    mask = ~result["CLOSEDATE"].isna() & (
        result["WHEN_TIMESTAMP"] + window >= result["CLOSEDATE"]
    )
    result.loc[mask, "target"] = 1

    # Get the total number of rows before filtering
    initial_row_count = len(result)

    # List to collect rows to keep
    rows_to_keep = []

    # Process each ID group
    for id_val, group in result.groupby("id"):
        # Check if this ID has any positive targets
        if 1 in group["target"].values:
            # Find the first occurrence of target = 1
            first_positive_idx = group[group["target"] == 1].index[0]

            # Keep all rows up to and including the first positive target
            rows_to_keep.extend(group.loc[:first_positive_idx].index.tolist())
        else:
            # If no positive targets, keep all rows for this ID
            rows_to_keep.extend(group.index.tolist())

    # Filter to keep only the selected rows
    filtered_result = result.loc[rows_to_keep].copy()

    # Count how many rows were assigned to each target value after filtering
    target_counts = filtered_result["target"].value_counts()
    logger.info(f"Target distribution: {target_counts.to_dict()}")

    # Calculate percentage of positive class
    positive_pct = (target_counts.get(1, 0) / len(filtered_result)) * 100
    logger.info(f"Positive class percentage: {positive_pct:.2f}%")

    # Log the filtering results
    rows_dropped = initial_row_count - len(filtered_result)
    logger.info(
        f"Rows dropped: {rows_dropped} ({rows_dropped / initial_row_count:.2%} of original data)"
    )
    logger.info(f"IDs after filtering: {filtered_result['id'].nunique()}")

    return filtered_result


def map_industry_to_bucket(industry_value: str) -> str:
    """
    Map an industry value to a standardized bucket using BUCKET_MAP.

    If the industry value isn't found in any bucket, return the original value.

    Args:
        industry_value: The raw industry value to map

    Returns:
        The standardized industry bucket or the original value if not found
    """
    # Handle None/NaN values
    if pd.isna(industry_value) or industry_value is None:
        return "Unknown"

    # Lowercase for case-insensitive matching
    industry_lower = industry_value.lower()

    # Search through each bucket in BUCKET_MAP
    for bucket, values in BUCKET_MAP.items():
        # Check if lowercase industry value is in the bucket
        if any(val.lower() == industry_lower for val in values):
            return bucket

    # If not found in any bucket, return the original value
    return industry_value


def transform_alexa_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the ALEXA_RANK feature with min-max scaling.

    For ALEXA_RANK, 1 is the best value and 16,000,001 is the worst value.
    This function creates a new feature 'alexa_rank_score' where:
    - A value of 1 (best rank) becomes 1.0
    - A value of 16,000,001 (worst rank) becomes 0.0
    - Missing values are filled with 0.0 (worst score)

    Args:
        df: DataFrame containing the ALEXA_RANK column

    Returns:
        DataFrame with the added alexa_rank_score column
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Set min and max values for ALEXA_RANK
    min_rank = 1  # Best ranking
    max_rank = 16000001  # Worst ranking

    # Calculate the min-max scaled score
    # Invert the scale so that lower ranks (better) get higher scores
    if "ALEXA_RANK" in result.columns:
        # First fill missing values with max_rank (worst rank)
        result["ALEXA_RANK"] = result["ALEXA_RANK"].fillna(max_rank)

        # Apply min-max scaling (and invert, so 1 is best)
        result["alexa_rank_score"] = 1 - (
            (result["ALEXA_RANK"] - min_rank) / (max_rank - min_rank)
        )

        # Clip values to ensure they're in the [0, 1] range
        result["alexa_rank_score"] = result["alexa_rank_score"].clip(0, 1)

        logger.info("Applied min-max scaling to ALEXA_RANK")
    else:
        logger.warning(
            "ALEXA_RANK column not found in DataFrame, skipping transformation"
        )

    return result


def calculate_cumulative_features(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    periods: int = 2,
    id_col: str = "id",
    timestamp_col: str = "WHEN_TIMESTAMP",
    fill_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate cumulative sums for numeric columns over specified time periods.

    This function calculates the sum of values over a specified time window for each account,
    including the current period. For example, with periods=2, it calculates the sum of
    the current week and the previous week's values.

    Args:
        df: DataFrame with time series data, must be sorted by id and timestamp
        numeric_cols: List of numeric column names to calculate cumulative features for.
                     If None, will detect all numeric columns except id and timestamp
        periods: Number of periods to include in the sum (including current period)
                Must be at least 2 (e.g., 2 for two-week total, 4 for four-week total)
        id_col: Name of the ID column that identifies unique accounts
        timestamp_col: Name of the timestamp column
        fill_method: Method to use for filling NaN values after calculation
                    (None, 'ffill', 'bfill', or a numeric value)

    Returns:
        DataFrame with added columns for cumulative values
    """
    if periods < 2:
        raise ValueError(
            "Periods must be at least 2 for cumulative calculations"
        )

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Ensure data is sorted by id and timestamp
    result = result.sort_values(by=[id_col, timestamp_col])

    # If no columns specified, find all numeric columns excluding id and timestamp
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in result.select_dtypes(include=["number"]).columns
            if col != id_col and col != timestamp_col
        ]

    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in numeric_cols if col in result.columns]

    if not valid_cols:
        logger.warning(
            "No valid numeric columns found for cumulative calculation"
        )
        return result

    # Log which columns we're calculating cumulative values for
    logger.info(
        f"Calculating {periods}-period cumulative values for columns: {valid_cols}"
    )

    # Count unique accounts
    num_accounts = result[id_col].nunique()
    logger.info(
        f"Processing cumulative values for {num_accounts} unique accounts"
    )

    # Generate column names for cumulative columns
    cumulative_cols = []

    # Process each account group separately
    for current_id, group in result.groupby(id_col):
        # Skip accounts with too few data points
        if len(group) < periods:
            # Fill with NaNs (will be handled later)
            for col in valid_cols:
                cum_col = f"{col}_cum_{periods}week"
                result.loc[group.index, cum_col] = None
                if cum_col not in cumulative_cols:
                    cumulative_cols.append(cum_col)
            continue

        # Calculate cumulative values for each valid column
        for col in valid_cols:
            cum_col = f"{col}_cum_{periods}week"

            # Initialize array for cumulative values
            cumulative_values = pd.Series(index=group.index)

            # For each row, calculate sum of this row and (periods-1) previous rows
            for i, (idx, row) in enumerate(group.iterrows()):
                if i < periods - 1:
                    # Not enough previous periods, use NaN
                    cumulative_values[idx] = None
                else:
                    # Sum current and (periods-1) previous values
                    window_values = group.iloc[i - (periods - 1) : i + 1][
                        col
                    ].values
                    cumulative_values[idx] = sum(window_values)

            # Add the cumulative values to the main DataFrame
            result.loc[group.index, cum_col] = cumulative_values

            # Keep track of new column names
            if cum_col not in cumulative_cols:
                cumulative_cols.append(cum_col)

    # Handle NaN values if a fill method is specified
    if fill_method is not None:
        if fill_method in ["ffill", "bfill"]:
            # Forward or backward fill
            result[cumulative_cols] = result[cumulative_cols].fillna(
                method=fill_method
            )
        else:
            # Try to convert to numeric and use as fill value
            try:
                fill_value = float(fill_method)
                result[cumulative_cols] = result[cumulative_cols].fillna(
                    fill_value
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid fill_method: {fill_method}, leaving NaN values"
                )

    # Log statistics about created columns
    for col in cumulative_cols:
        non_null_count = result[col].count()
        non_null_pct = non_null_count / len(result) * 100
        logger.info(
            f"  - {col}: {non_null_count} non-null values ({non_null_pct:.2f}%)"
        )

    return result


def calculate_percentage_change(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    periods: int = 1,
    id_col: str = "id",
    timestamp_col: str = "WHEN_TIMESTAMP",
    fill_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate percentage change in numeric columns over specified time periods.

    This function calculates percentage change for each account/ID separately,
    ensuring changes are not calculated across different accounts. It handles
    cases where there isn't enough history (e.g., only one data point per account)
    and avoids division by zero by replacing zeros with ones before calculation.

    Args:
        df: DataFrame with time series data, must be sorted by id and timestamp
        numeric_cols: List of numeric column names to calculate percentage change for.
                     If None, will detect all numeric columns except id and timestamp
        periods: Number of periods to shift for calculating percentage change
                (e.g., 1 for week-over-week, 2 for 2-week change)
        id_col: Name of the ID column that identifies unique accounts
        timestamp_col: Name of the timestamp column
        fill_method: Method to use for filling NaN values after calculation
                    (None, 'ffill', 'bfill', or a numeric value)

    Returns:
        DataFrame with added columns for percentage changes
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Ensure data is sorted by id and timestamp
    result = result.sort_values(by=[id_col, timestamp_col])

    # If no columns specified, find all numeric columns excluding id and timestamp
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in result.select_dtypes(include=["number"]).columns
            if col != id_col and col != timestamp_col
        ]

    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in numeric_cols if col in result.columns]

    if not valid_cols:
        logger.warning(
            "No valid numeric columns found for percentage change calculation"
        )
        return result

    # Log which columns we're calculating percentage change for
    logger.info(
        f"Calculating {periods}-period percentage change for columns: {valid_cols}"
    )

    # Count unique accounts
    num_accounts = result[id_col].nunique()
    logger.info(
        f"Processing percentage changes for {num_accounts} unique accounts"
    )

    # Generate column names for percentage change columns
    pct_change_cols = []

    # Process each account group separately to avoid calculating changes across different accounts
    for current_id, group in result.groupby(id_col):
        # Skip accounts with only one data point if periods >= 1
        if len(group) <= periods:
            # Fill with NaNs (will be handled later)
            for col in valid_cols:
                pct_col = f"{col}_pct_change_{periods}week"
                result.loc[group.index, pct_col] = None
                if pct_col not in pct_change_cols:
                    pct_change_cols.append(pct_col)
            continue

        # Calculate percentage change for each valid column
        for col in valid_cols:
            pct_col = f"{col}_pct_change_{periods}week"

            # Handle division by zero by custom percentage change calculation
            group_col = group[col].copy()
            shifted = group_col.shift(periods)

            # Replace zeros with ones in the denominator to avoid division by zero
            shifted_no_zeros = shifted.replace(0, 1)

            # Calculate percentage change: (current - previous) / previous
            pct_change = (group_col - shifted) / shifted_no_zeros

            # Add the result back to the main DataFrame
            result.loc[group.index, pct_col] = pct_change

            # Keep track of new column names
            if pct_col not in pct_change_cols:
                pct_change_cols.append(pct_col)

    # Handle NaN values if a fill method is specified
    if fill_method is not None:
        if fill_method in ["ffill", "bfill"]:
            # Forward or backward fill
            result[pct_change_cols] = result[pct_change_cols].fillna(
                method=fill_method
            )
        else:
            # Try to convert to numeric and use as fill value
            try:
                fill_value = float(fill_method)
                result[pct_change_cols] = result[pct_change_cols].fillna(
                    fill_value
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid fill_method: {fill_method}, leaving NaN values"
                )

    # Log statistics about created columns
    for col in pct_change_cols:
        non_null_count = result[col].count()
        non_null_pct = non_null_count / len(result) * 100
        logger.info(
            f"  - {col}: {non_null_count} non-null values ({non_null_pct:.2f}%)"
        )

        # Check for any infinite values that might have slipped through
        inf_count = result[col].isin([float("inf"), -float("inf")]).sum()
        if inf_count > 0:
            logger.warning(
                f"  - {col}: {inf_count} infinite values found - this shouldn't happen with our approach"
            )

    return result


def calculate_cumulative_to_date(
    df: pd.DataFrame,
    numeric_cols: List[str] = None,
    id_col: str = "id",
    timestamp_col: str = "WHEN_TIMESTAMP",
    fill_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate cumulative totals to date for numeric columns.

    This function calculates the running sum of values from the beginning of time
    for each account up to the current data point. It gives the total accumulated
    metric for each account at each timestamp.

    Args:
        df: DataFrame with time series data, must be sorted by id and timestamp
        numeric_cols: List of numeric column names to calculate cumulative features for.
                     If None, will detect all numeric columns except id and timestamp
        id_col: Name of the ID column that identifies unique accounts
        timestamp_col: Name of the timestamp column
        fill_method: Method to use for filling NaN values after calculation
                    (None, 'ffill', 'bfill', or a numeric value)

    Returns:
        DataFrame with added columns for cumulative to date values
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Ensure data is sorted by id and timestamp
    result = result.sort_values(by=[id_col, timestamp_col])

    # If no columns specified, find all numeric columns excluding id and timestamp
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in result.select_dtypes(include=["number"]).columns
            if col != id_col and col != timestamp_col
        ]

    # Filter to only include columns that exist in the DataFrame
    valid_cols = [col for col in numeric_cols if col in result.columns]

    if not valid_cols:
        logger.warning(
            "No valid numeric columns found for cumulative to date calculation"
        )
        return result

    # Log which columns we're calculating cumulative values for
    logger.info(
        f"Calculating cumulative to date values for columns: {valid_cols}"
    )

    # Count unique accounts
    num_accounts = result[id_col].nunique()
    logger.info(
        f"Processing cumulative to date values for {num_accounts} unique accounts"
    )

    # Generate column names for cumulative columns
    cumulative_cols = []

    # Process each account group separately
    for current_id, group in result.groupby(id_col):
        # Calculate cumulative to date values for each valid column
        for col in valid_cols:
            cum_col = f"{col}_cum_to_date"

            # Calculate cumulative sum for this account
            # This creates a running total from the first record to the current record
            cumulative_values = group[col].cumsum()

            # Add the cumulative values to the main DataFrame
            result.loc[group.index, cum_col] = cumulative_values

            # Keep track of new column names
            if cum_col not in cumulative_cols:
                cumulative_cols.append(cum_col)

    # Handle NaN values if a fill method is specified
    if fill_method is not None:
        if fill_method in ["ffill", "bfill"]:
            # Forward or backward fill
            result[cumulative_cols] = result[cumulative_cols].fillna(
                method=fill_method
            )
        else:
            # Try to convert to numeric and use as fill value
            try:
                fill_value = float(fill_method)
                result[cumulative_cols] = result[cumulative_cols].fillna(
                    fill_value
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid fill_method: {fill_method}, leaving NaN values"
                )

    # Log statistics about created columns
    for col in cumulative_cols:
        non_null_count = result[col].count()
        non_null_pct = non_null_count / len(result) * 100
        logger.info(
            f"  - {col}: {non_null_count} non-null values ({non_null_pct:.2f}%)"
        )

    return result


def transform_industry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the INDUSTRY column by mapping raw values to standardized buckets.

    This function creates a new column 'industry_bucket' with standardized values
    based on the mapping defined in BUCKET_MAP.

    Args:
        df: DataFrame containing the INDUSTRY column

    Returns:
        DataFrame with the added industry_bucket column
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    if "INDUSTRY" in result.columns:
        # Apply the mapping function to each value in the INDUSTRY column
        result["industry_bucket"] = result["INDUSTRY"].apply(
            map_industry_to_bucket
        )

        # Count how many values were mapped to each bucket
        bucket_counts = result["industry_bucket"].value_counts()

        # Log summary statistics
        total_values = len(result)
        mapped_count = sum(
            count
            for bucket, count in bucket_counts.items()
            if bucket in BUCKET_MAP
        )
        unmapped_count = total_values - mapped_count

        logger.info(
            f"Mapped {mapped_count} industry values to standardized buckets"
        )
        if unmapped_count > 0:
            unmapped_pct = (unmapped_count / total_values) * 100
            logger.warning(
                f"{unmapped_count} industry values ({unmapped_pct:.2f}%) could not be mapped to buckets"
            )

        # Log the distribution of industry buckets
        logger.info("Industry bucket distribution:")
        for bucket, count in bucket_counts.items():
            if bucket in BUCKET_MAP:
                logger.info(
                    f"  - {bucket}: {count} ({count / total_values:.2%})"
                )
    else:
        logger.warning(
            "INDUSTRY column not found in DataFrame, skipping transformation"
        )

    return result


def apply_feature_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations to the DataFrame.

    Args:
        df: DataFrame with original features

    Returns:
        DataFrame with transformed features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Apply ALEXA_RANK transformation
    result = transform_alexa_rank(result)

    # Apply INDUSTRY transformation
    result = transform_industry(result)

    # Define the usage columns for transformations
    numeric_usage_cols = [
        "ACTIONS_CRM_CONTACTS",
        "ACTIONS_CRM_COMPANIES",
        "ACTIONS_CRM_DEALS",
        "ACTIONS_EMAIL",
        "USERS_CRM_CONTACTS",
        "USERS_CRM_COMPANIES",
        "USERS_CRM_DEALS",
        "USERS_EMAIL",
    ]

    # Calculate week-over-week percentage changes (periods=1)
    result = calculate_percentage_change(
        result,
        numeric_cols=numeric_usage_cols,
        periods=1,  # Week-over-week changes
    )

    # Calculate two-week percentage changes (periods=2)
    result = calculate_percentage_change(
        result,
        numeric_cols=numeric_usage_cols,
        periods=2,  # 2-week changes
    )

    # Calculate cumulative features for different time periods

    # 2-week cumulative (current week + previous week)
    result = calculate_cumulative_features(
        result,
        numeric_cols=numeric_usage_cols,
        periods=2,
    )

    # 4-week cumulative (current week + previous 3 weeks)
    result = calculate_cumulative_features(
        result,
        numeric_cols=numeric_usage_cols,
        periods=4,
    )

    # Calculate cumulative to date (total accumulation from the beginning)
    result = calculate_cumulative_to_date(
        result,
        numeric_cols=numeric_usage_cols,
    )

    # Log the result
    logger.info(f"Applied feature transformations to {len(result)} rows")

    return result


def process_features_and_target(
    data_path: Optional[str | Path] = None, window_days: int = 28
) -> pd.DataFrame:
    """
    Process the data to create features and target variables.

    Args:
        data_path: Path to the processed data file
        window_days: Number of days to look ahead for conversion

    Returns:
        DataFrame with features and target variable
    """
    # Load the processed data
    data = load_processed_data(data_path)

    # Prepare data for feature engineering
    sorted_data = prepare_data_for_features(data)

    # Create target variable
    data_with_target = create_target_variable(sorted_data, window_days)

    # Apply feature transformations
    data_with_features = apply_feature_transformations(data_with_target)

    # Log the result
    logger.info(
        f"Created features and target for {len(data_with_features)} rows"
    )
    logger.info(f"Unique accounts: {data_with_features['id'].nunique()}")

    return data_with_features


if __name__ == "__main__":
    # Example usage
    result = process_features_and_target()
    logger.info(f"Processed data shape: {result.shape}")
    logger.info(f"Sample data with target:\n{result.head()}")
