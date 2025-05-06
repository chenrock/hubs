"""
Data transformation module for the Hubs project.

This module provides functions to transform and join the raw data files into
a single dataset that can be used for modeling.
"""

import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean duplicate ID rows in a DataFrame.

    For rows with duplicate IDs:
    1. Verify all non-null values are consistent across duplicates
    2. Merge duplicate rows to get a single row with all non-null values
    3. If there are conflicting non-null values, exclude all rows with that ID

    Args:
        df: DataFrame to clean

    Returns:
        DataFrame with duplicates cleaned
    """
    # Check if there are any duplicate IDs
    duplicate_ids = df["id"].duplicated(keep=False)
    if not duplicate_ids.any():
        return df  # No duplicates to clean

    # Separate the duplicates and non-duplicates
    duplicates = df[duplicate_ids].copy()
    non_duplicates = df[~duplicate_ids].copy()

    # Get unique duplicate IDs
    unique_duplicate_ids = duplicates["id"].unique()

    # Create a list to store the cleaned rows
    cleaned_rows = []
    # Store IDs to exclude due to conflicts
    exclude_ids = set()

    # Process each set of duplicate rows
    for dup_id in unique_duplicate_ids:
        dup_rows = duplicates[duplicates["id"] == dup_id]

        # Create a merged row
        merged_row = {}
        has_conflict = False

        # Start with the ID
        merged_row["id"] = dup_id

        # Process each column
        for col in df.columns:
            if col == "id":  # Skip ID column since we already set it
                continue

            # Get non-null values for this column
            non_null_values = dup_rows[col].dropna().unique()

            if len(non_null_values) == 0:
                # All values are null
                merged_row[col] = None
            elif len(non_null_values) == 1:
                # Only one unique non-null value, use it
                merged_row[col] = non_null_values[0]
            else:
                # Multiple different non-null values - conflict
                logger.warning(
                    f"Conflicting values for id={dup_id}, column={col}: {non_null_values}"
                )
                # Mark this ID for exclusion
                has_conflict = True
                exclude_ids.add(dup_id)
                break

        if not has_conflict:
            cleaned_rows.append(merged_row)

    # Convert cleaned rows to DataFrame
    if cleaned_rows:
        cleaned_duplicates = pd.DataFrame(cleaned_rows)
        # Combine non-duplicates with cleaned duplicates
        cleaned_df = pd.concat(
            [non_duplicates, cleaned_duplicates], ignore_index=True
        )
    else:
        cleaned_df = non_duplicates.copy()

    # Log summary
    num_excluded = len(exclude_ids)
    if num_excluded > 0:
        logger.info(
            f"Excluded {num_excluded} IDs due to conflicting values: {exclude_ids}"
        )

    return cleaned_df


def load_raw_data(
    data_dir: str | Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV data files.

    Args:
        data_dir: Path to the directory containing the raw data files

    Returns:
        A tuple of (customers, noncustomers, usage_actions) DataFrames
    """
    if data_dir is None:
        # Use path from project root
        data_dir = Path(".", "data", "raw")
    else:
        data_dir = Path(data_dir)

    # Load the data files
    customers = pd.read_csv(data_dir / "customers_(4).csv")
    noncustomers = pd.read_csv(data_dir / "noncustomers_(4).csv")
    usage_actions = pd.read_csv(data_dir / "usage_actions_(4).csv")

    # Convert date columns to datetime
    customers["CLOSEDATE"] = pd.to_datetime(customers["CLOSEDATE"])
    usage_actions["WHEN_TIMESTAMP"] = pd.to_datetime(
        usage_actions["WHEN_TIMESTAMP"]
    )

    return customers, noncustomers, usage_actions


def create_accounts_table(
    customers: pd.DataFrame, noncustomers: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a unified accounts table by joining customers and noncustomers.

    For noncustomers, CLOSEDATE and MRR will be set to null.

    Args:
        customers: DataFrame containing customer data
        noncustomers: DataFrame containing noncustomer data

    Returns:
        A unified accounts DataFrame
    """
    # Add CLOSEDATE and MRR columns to noncustomers with null values
    noncustomers_with_nulls = noncustomers.copy()

    # Create datetime CLOSEDATE column with NaT values (null for datetime)
    noncustomers_with_nulls["CLOSEDATE"] = pd.NaT

    # Create MRR column with appropriate dtype (float64 to match customers)
    noncustomers_with_nulls["MRR"] = pd.Series(dtype=customers["MRR"].dtype)

    # Ensure all columns have matching dtypes for clean concatenation
    for col in customers.columns:
        if col in noncustomers_with_nulls and col not in ["CLOSEDATE", "MRR"]:
            # Make sure dtypes match between the two DataFrames
            noncustomers_with_nulls[col] = noncustomers_with_nulls[col].astype(
                customers[col].dtype, errors="ignore"
            )

    # Reorder noncustomers columns to match customers
    noncustomers_with_nulls = noncustomers_with_nulls[customers.columns]

    # Concatenate the two DataFrames
    accounts = pd.concat(
        [customers, noncustomers_with_nulls], ignore_index=True
    )

    # Clean duplicate IDs
    logger.info("Clean data: dedup IDs")
    accounts = clean_duplicate_ids(accounts)

    return accounts


def join_accounts_with_usage(
    accounts: pd.DataFrame, usage_actions: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    """
    Join the accounts table with usage actions.

    Args:
        accounts: DataFrame containing unified account data
        usage_actions: DataFrame containing usage action data

    Returns:
        A tuple containing:
        - joined DataFrame with account and usage data (only accounts with activity)
        - List of account IDs that have no activity
    """
    # Perform a left join on accounts with usage_actions
    joined_data = accounts.merge(
        usage_actions,
        on="id",
        how="left",
        indicator=True,
    )

    # Identify accounts with no usage data (where _merge is "left_only")
    no_activity_mask = joined_data["_merge"] == "left_only"
    accounts_with_no_activity_ids = (
        joined_data[no_activity_mask]["id"].unique().tolist()
    )

    # Keep only rows where usage data exists
    joined_data_with_activity = joined_data[~no_activity_mask].copy()

    # Drop the _merge indicator column
    if "_merge" in joined_data_with_activity.columns:
        joined_data_with_activity = joined_data_with_activity.drop(
            columns=["_merge"]
        )

    return joined_data_with_activity, accounts_with_no_activity_ids


def process_data(data_dir: str | Path = None) -> pd.DataFrame:
    """
    Process the raw data files into a single joined dataset.

    Args:
        data_dir: Path to the directory containing the raw data files

    Returns:
        A processed DataFrame ready for feature engineering
    """
    # Load the raw data (cleaning duplicates in customers and noncustomers)
    customers, noncustomers, usage_actions = load_raw_data(data_dir)

    # Create the accounts table
    accounts = create_accounts_table(customers, noncustomers)

    # Join accounts with usage actions
    joined_data, accounts_with_no_activity = join_accounts_with_usage(
        accounts, usage_actions
    )

    # Log data sizes for verification
    logger.info(
        f"Customers: {len(customers)}, Noncustomers: {len(noncustomers)}, Usage actions: {len(usage_actions)}"
    )
    logger.info(
        f"Unique customer IDs: {customers['id'].nunique()}, Unique noncustomer IDs: {noncustomers['id'].nunique()}"
    )
    logger.info(
        f"Accounts after joining: {len(accounts)}, Unique account IDs: {accounts['id'].nunique()}"
    )

    # Log information about accounts with no activity
    num_accounts_no_activity = len(accounts_with_no_activity)
    logger.info(f"Accounts with no activity: {num_accounts_no_activity}")

    # Break down no-activity accounts by customer vs non-customer
    customer_ids = set(customers["id"].unique())
    no_activity_customers = [
        id for id in accounts_with_no_activity if id in customer_ids
    ]
    no_activity_noncustomers = [
        id for id in accounts_with_no_activity if id not in customer_ids
    ]

    logger.info(f"Customers with no activity: {len(no_activity_customers)}")
    logger.info(
        f"Non-customers with no activity: {len(no_activity_noncustomers)}"
    )

    # Log the actual IDs that have no activity (can be useful for debugging/alerting)
    if no_activity_customers:
        logger.info(f"Customer IDs with no activity: {no_activity_customers}")

    # Count customers and non-customers in the final joined data by unique ID
    customer_ids_in_joined = set(joined_data["id"].unique()) & customer_ids
    joined_customers = len(customer_ids_in_joined)
    joined_noncustomers = joined_data["id"].nunique() - joined_customers

    # Log data sizes for the final joined data
    logger.info(
        f"Final joined data: {len(joined_data)}, Unique IDs in joined data: {joined_data['id'].nunique()}"
    )
    logger.info(
        f"Customers in final data: {joined_customers}, Non-customers in final data: {joined_noncustomers}"
    )

    # reorder columns
    joined_data = joined_data[
        [
            "id",
            "WHEN_TIMESTAMP",
            "CLOSEDATE",
            "MRR",
            "ALEXA_RANK",
            "EMPLOYEE_RANGE",
            "INDUSTRY",
            "ACTIONS_CRM_CONTACTS",
            "ACTIONS_CRM_COMPANIES",
            "ACTIONS_CRM_DEALS",
            "ACTIONS_EMAIL",
            "USERS_CRM_CONTACTS",
            "USERS_CRM_COMPANIES",
            "USERS_CRM_DEALS",
            "USERS_EMAIL",
        ]
    ]

    return joined_data


if __name__ == "__main__":
    # Example usage
    processed_data = process_data()
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Sample data:\n{processed_data.head()}")
