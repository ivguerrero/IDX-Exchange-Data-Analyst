from pathlib import Path
 
import pandas as pd
 
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "CRMLSSold_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "CRMLSSold_features.csv"
 
DATE_COLS = [
    "CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate",
]
 

PRICE_RATIO_BASE = "OriginalListPrice"
 
NEW_COLS = [
    "price_ratio", "close_to_original_list_ratio", "price_per_sqft",
    "close_year", "close_month", "yr_mo",
    "listing_to_contract_days", "contract_to_close_days",
]
 
pd.options.display.float_format = "{:,.3f}".format
pd.options.display.width = 200
 
 
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def safe_ratio(numerator, denominator):
    return (numerator / denominator).where(denominator > 0)
 
 
# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def load_data():
    """Read the cleaned CSV. Dates come back from CSV as text, so convert them again."""
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"Loaded {INPUT_FILE.name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df
 
 
def add_metrics(df):
    """Create the engineered metric columns."""

    df["price_ratio"] = safe_ratio(df["ClosePrice"], df[PRICE_RATIO_BASE])
    df["close_to_original_list_ratio"] = safe_ratio(df["ClosePrice"], df["OriginalListPrice"])
 
    df["price_per_sqft"] = safe_ratio(df["ClosePrice"], df["LivingArea"])

    df["close_year"] = df["CloseDate"].dt.year
    df["close_month"] = df["CloseDate"].dt.month
    df["yr_mo"] = df["CloseDate"].dt.strftime("%Y-%m")

    df["listing_to_contract_days"] = (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days
    df["contract_to_close_days"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days
    return df
 
 
def report_metrics(df):
    """Confirm the new columns are populated and show their distributions."""
    print(f"\nprice_ratio uses: ClosePrice / {PRICE_RATIO_BASE}")
 
    print("\nPopulated counts (blank = denominator was zero/blank, or a date was missing)")
    for col in NEW_COLS:
        filled = df[col].notnull().sum()
        print(f"  {col}: {filled:,} of {len(df):,} filled ({len(df) - filled:,} blank)")
 
    numeric_new = [c for c in NEW_COLS if c not in ("yr_mo",)]
    summary = df[numeric_new].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    print("\nDistribution of the new numeric columns")
    print(summary.to_string())
 
    # Sanity checks worth reading: most ratios should sit near 1.0
    for col in ["price_ratio", "close_to_original_list_ratio"]:
        odd = ((df[col] < 0.5) | (df[col] > 2)).sum()
        print(f"\n{col}: {odd:,} values below 0.5 or above 2 (check these for bad prices)")
    for col in ["listing_to_contract_days", "contract_to_close_days"]:
        print(f"{col}: {(df[col] < 0).sum():,} negative values (dates out of order)")
 
 
def show_sample(df):
    """Print a sample table of the new columns next to the fields they come from."""
    cols = ["CloseDate", "ClosePrice", "OriginalListPrice", "ListPrice", "LivingArea",
            "DaysOnMarket"] + NEW_COLS
    print("\nSample output (first 10 rows)")
    print(df[cols].head(10).to_string())
 
 
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_data()
    df = add_metrics(df)
    report_metrics(df)
    show_sample(df)
 
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df):,} rows, {df.shape[1]} columns to {OUTPUT_FILE.name}")
 
 
if __name__ == "__main__":
    main()
