"""
validate.py - Weeks 2-3 dataset structuring and validation (EDA)

Works on the combined Residential Sold file:
  1. Load the data and convert date columns
  2. (Optional) missing-value report on all columns, flagging > 90% missing
  3. Drop the high-missing columns
  4. Percentile summary for the nine key numeric fields
  5. Histograms (1st-99th percentile) and boxplots (full data) for those fields
  6. Intern questions: close price, days on market, sold above/below list,
     date consistency, median price by county

Run from anywhere; paths are built from this script's own folder, so the
script and the combined CSV are expected to sit together in the csv folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "CRMLSSold_combined_residential.csv"
PLOT_DIR = BASE_DIR / "plots"

# Switch sections on or off
RUN_MISSING_REPORT = False   # long printout (one line per column), so off by default
RUN_SUMMARY = True
RUN_HISTOGRAMS = True
RUN_BOXPLOTS = True
RUN_QUESTIONS = True

MIN_COUNTY_SALES = 100       # counties with fewer sales are left out of the county ranking

MISSING_FLAG_PCT = 90
PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

pd.options.display.float_format = "{:,.2f}".format

# ----------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------
DATE_COLS = [
    "CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate",
]

# Dropped: 27 columns above 65% missing (all 15 flagged at 90% plus 12 more)
DROP_COLS = [
    # Above 90% missing (15)
    "FireplacesTotal", "AboveGradeFinishedArea", "TaxAnnualAmount", "TaxYear",
    "ElementarySchoolDistrict", "BusinessType", "CoveredSpaces",
    "MiddleOrJuniorSchoolDistrict", "WaterfrontYN", "BelowGradeFinishedArea",
    "BasementYN", "BuilderName", "LotSizeDimensions", "BuildingAreaTotal",
    "CoBuyerAgentFirstName",
    # 65-90% missing (12)
    "BuyerAgencyCompensationType", "BuyerAgencyCompensation", "ElementarySchool",
    "MiddleOrJuniorSchool", "HighSchool", "latfilled", "lonfilled",
    "CoListAgentFirstName", "CoListAgentLastName", "CoListOfficeName",
    "OriginatingSystemName", "OriginatingSystemSubName",
]

# Market analysis fields (32)
MARKET_COLS = [
    "OriginalListPrice", "ListPrice", "ClosePrice",
    "CloseDate", "ListingContractDate", "PurchaseContractDate", "DaysOnMarket",
    "LivingArea", "LotSizeAcres", "LotSizeArea", "LotSizeSquareFeet",
    "BedroomsTotal", "BathroomsTotalInteger", "MainLevelBedrooms", "YearBuilt", "Stories",
    "GarageSpaces", "ParkingTotal", "AttachedGarageYN", "PoolPrivateYN",
    "ViewYN", "FireplaceYN", "Flooring", "NewConstructionYN",
    "CountyOrParish", "City", "PostalCode", "Latitude", "Longitude", "StateOrProvince",
    "AssociationFee", "AssociationFeeFrequency",
]

# Metadata fields (25)
METADATA_COLS = [
    "ListingKey", "ListingKeyNumeric", "ListingId",
    "ListAgentEmail", "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "BuyerAgentMlsId", "BuyerAgentFirstName", "BuyerAgentLastName",
    "ListOfficeName", "BuyerOfficeName", "BuyerOfficeAOR", "BuyerAgentAOR", "ListAgentAOR",
    "UnparsedAddress", "StreetNumberNumeric", "MLSAreaMajor", "SubdivisionName",
    "HighSchoolDistrict", "PropertyType", "MlsStatus", "PropertySubType", "Levels",
    "ContractStatusChangeDate",
]

# Numeric fields for the distribution review
NUM_COLS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]
WHOLE_NUMBER_COLS = ["BedroomsTotal", "BathroomsTotalInteger"]   # one histogram bin per integer
LOG_BOX_COLS = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres"]


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def load_data():
    """Read the combined CSV and convert the date columns to real dates."""
    df = pd.read_csv(DATA_FILE, low_memory=False)
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def report_missing(df):
    """Print missing count and percent for every column; flag those above the cutoff."""
    for col, count in df.isnull().sum().items():
        pct = count / len(df) * 100
        print(f"{col} : {count} missing. {pct:.1f}%")
        if pct > MISSING_FLAG_PCT:
            print("*****************FLAGGED********************")


def percentile_summary(df):
    """Print count, mean, std, min, percentiles, and max for the numeric fields."""
    summary = df[NUM_COLS].describe(percentiles=PERCENTILES).T
    print(summary.to_string())


def plot_histograms(df):
    """One histogram per numeric field, restricted to the 1st-99th percentile."""
    for col in NUM_COLS:
        data = df[col].dropna()
        lo, hi = data.quantile([0.01, 0.99])
        trimmed = data[(data >= lo) & (data <= hi)]

        # One bin per whole number for bed/bath counts, 50 bins otherwise
        bins = np.arange(lo - 0.5, hi + 1.5, 1) if col in WHOLE_NUMBER_COLS else 50

        plt.figure(figsize=(8, 5))
        plt.hist(trimmed, bins=bins, edgecolor="black")
        plt.title(f"{col} (1st to 99th percentile)")
        plt.xlabel(col)
        plt.ylabel("Number of sales")
        plt.savefig(PLOT_DIR / f"{col}_hist.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"{col}: kept {len(trimmed):,} of {len(data):,} values ({lo:,.2f} to {hi:,.2f})")


def plot_boxplots(df):
    """One boxplot per numeric field on the full data, plus a log-scale version for skewed fields."""
    for col in NUM_COLS:
        data = df[col].dropna()   # no trimming: the outliers are the point

        # Outlier count using the 1.5 x IQR rule
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((data < low_fence) | (data > high_fence)).sum()
        print(f"{col}: {n_out:,} outliers ({n_out / len(data) * 100:.1f}%), "
              f"fences {low_fence:,.2f} to {high_fence:,.2f}")

        plt.figure(figsize=(8, 4))
        plt.boxplot(data, vert=False, flierprops={"markersize": 2})
        plt.title(f"{col} (full data)")
        plt.xlabel(col)
        plt.savefig(PLOT_DIR / f"{col}_box.png", dpi=150, bbox_inches="tight")
        plt.close()

        if col in LOG_BOX_COLS:
            positive = data[data > 0]   # a log axis can't show zeros or negatives
            plt.figure(figsize=(8, 4))
            plt.boxplot(positive, vert=False, flierprops={"markersize": 2})
            plt.xscale("log")
            plt.title(f"{col} (full data, log scale, values > 0)")
            plt.xlabel(f"{col} (log scale)")
            plt.savefig(PLOT_DIR / f"{col}_box_log.png", dpi=150, bbox_inches="tight")
            plt.close()


def q1_close_price(df):
    """Q1: median and mean close price (rows with a $0 or blank close price are excluded)."""
    prices = df.loc[df["ClosePrice"] > 0, "ClosePrice"]
    print(f"Sales with a valid close price: {len(prices):,} of {len(df):,}")
    print(f"Median close price: ${prices.median():,.0f}")
    print(f"Mean close price:   ${prices.mean():,.0f}")


def q2_days_on_market(df):
    """Q2: days on market distribution, plus how many values look invalid."""
    dom = df["DaysOnMarket"].dropna()
    print(dom.describe(percentiles=PERCENTILES).to_string())
    print(f"Negative values: {(dom < 0).sum():,}")
    print(f"Zero days:       {(dom == 0).sum():,}")
    print(f"Over 365 days:   {(dom > 365).sum():,}")


def q3_above_below_list(df):
    """Q3: share of sales that closed above, at, or below list price.
    Rows with a blank or non-positive close or list price are excluded."""
    for list_col in ["ListPrice", "OriginalListPrice"]:
        valid = df[(df["ClosePrice"] > 0) & (df[list_col] > 0)]
        above = (valid["ClosePrice"] > valid[list_col]).mean() * 100
        below = (valid["ClosePrice"] < valid[list_col]).mean() * 100
        equal = (valid["ClosePrice"] == valid[list_col]).mean() * 100
        print(f"Close vs {list_col} ({len(valid):,} sales): "
              f"above {above:.1f}%, at {equal:.1f}%, below {below:.1f}%")


def q4_date_consistency(df):
    """Q4: count rows where dates fall in an impossible order.
    Rows with a blank date in the comparison are left out of that check."""
    checks = {
        "CloseDate before ListingContractDate": df["CloseDate"] < df["ListingContractDate"],
        "PurchaseContractDate before ListingContractDate": df["PurchaseContractDate"] < df["ListingContractDate"],
        "CloseDate before PurchaseContractDate": df["CloseDate"] < df["PurchaseContractDate"],
    }
    for label, bad in checks.items():
        print(f"{label}: {bad.sum():,} rows ({bad.mean() * 100:.2f}% of all rows)")

    negative_dom = df["DaysOnMarket"] < 0
    overlap = (negative_dom & checks["CloseDate before ListingContractDate"]).sum()
    print(f"Negative DaysOnMarket rows: {negative_dom.sum():,} "
          f"({overlap:,} of them also have CloseDate before ListingContractDate)")


def q5_county_medians(df):
    """Q5: median close price by county, highest first, with the sale count shown."""
    valid = df[df["ClosePrice"] > 0]
    by_county = valid.groupby("CountyOrParish")["ClosePrice"].agg(["median", "count"])
    ranked = by_county[by_county["count"] >= MIN_COUNTY_SALES].sort_values("median", ascending=False)
    print(f"Counties with at least {MIN_COUNTY_SALES} sales: {len(ranked)} of {len(by_county)}")
    print(ranked.head(10).to_string())


def answer_questions(df):
    """Run the intern questions in order."""
    for title, func in [
        ("Q1: Median and mean close price", q1_close_price),
        ("Q2: Days on market distribution", q2_days_on_market),
        ("Q3: Sold above vs. below list price", q3_above_below_list),
        ("Q4: Date consistency", q4_date_consistency),
        ("Q5: Highest median prices by county", q5_county_medians),
    ]:
        print(f"\n=== {title} ===")
        func(df)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    sold = load_data()

    # The missing-value report runs on all columns, before anything is dropped
    if RUN_MISSING_REPORT:
        report_missing(sold)

    sold = sold.drop(columns=DROP_COLS)
    print(f"Shape after dropping {len(DROP_COLS)} columns: {sold.shape}")

    # Every remaining column should be in exactly one of the two lists
    unplaced = set(sold.columns) - set(MARKET_COLS) - set(METADATA_COLS)
    print(f"Columns not in market or metadata lists: {unplaced or 'none'}")

    PLOT_DIR.mkdir(exist_ok=True)

    if RUN_SUMMARY:
        percentile_summary(sold)
    if RUN_HISTOGRAMS:
        plot_histograms(sold)
    if RUN_BOXPLOTS:
        plot_boxplots(sold)
    if RUN_QUESTIONS:
        answer_questions(sold)


if __name__ == "__main__":
    main()