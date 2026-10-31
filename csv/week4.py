from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "CRMLSSold_enriched.csv"
OUTPUT_FILE = BASE_DIR / "CRMLSSold_cleaned.csv"
INVALID_COORDS_FILE = BASE_DIR / "CRMLSSold_invalid_coordinates.csv"

DATE_COLS = [
    "CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate",
]

# Columns that must be numeric
NUMERIC_COLS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
    "Latitude", "Longitude", "rate_30yr_fixed",
]

# Dropped
DROP_COLS = [
    # Above 90% missing
    "FireplacesTotal", "AboveGradeFinishedArea", "TaxAnnualAmount", "TaxYear",
    "ElementarySchoolDistrict", "BusinessType", "CoveredSpaces",
    "MiddleOrJuniorSchoolDistrict", "WaterfrontYN", "BelowGradeFinishedArea",
    "BasementYN", "BuilderName", "LotSizeDimensions", "BuildingAreaTotal",
    "CoBuyerAgentFirstName",
    # 65-90% missing
    "BuyerAgencyCompensationType", "BuyerAgencyCompensation", "ElementarySchool",
    "MiddleOrJuniorSchool", "HighSchool",
    "CoListAgentFirstName", "CoListAgentLastName", "CoListOfficeName",
    "OriginatingSystemName", "OriginatingSystemSubName",
]

# Only present in some monthly files. Kept until the geographic check has
# looked at them, then dropped, because Latitude/Longitude are the
# coordinates used for analysis.
COORD_HELPER_COLS = ["latfilled", "lonfilled"]

# Approximate bounding box for California. Coordinates outside it are
# flagged as out-of-state or implausible. Worth confirming the exact bounds.
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
CA_LON_MIN, CA_LON_MAX = -124.5, -114.1

row_log = []   # (step, row count) after each step, for the before/after table


def log_rows(step, df):
    row_log.append((step, len(df)))
    print(f"[rows] {step}: {len(df):,}")


# ----------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {INPUT_FILE.name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    log_rows("Loaded", df)
    return df


def convert_types(df):
    for col in DATE_COLS:
        before = df[col].isnull().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        after = df[col].isnull().sum()
        print(f"{col}: blanks {before:,} -> {after:,} after datetime conversion")

    for col in NUMERIC_COLS:
        before = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isnull().sum()
        if after != before:
            print(f"{col}: blanks {before:,} -> {after:,} after numeric conversion")
    return df


def drop_columns(df):
    cols_before = df.shape[1]
    df = df.drop(columns=DROP_COLS)
    print(f"Dropped {len(DROP_COLS)} columns: {cols_before} -> {df.shape[1]}")
    return df


def remove_invalid_rows(df):
    rules = [
        ("ClosePrice missing or <= 0", lambda d: ~(d["ClosePrice"] > 0)),
        ("LivingArea <= 0", lambda d: d["LivingArea"] <= 0),
        ("DaysOnMarket < 0", lambda d: d["DaysOnMarket"] < 0),
        ("Negative bedrooms or bathrooms",
         lambda d: (d["BedroomsTotal"] < 0) | (d["BathroomsTotalInteger"] < 0)),
    ]
    for label, rule in rules:
        mask = rule(df)
        print(f"Removing {mask.sum():,} rows: {label}")
        df = df[~mask]
        log_rows(f"After removing: {label}", df)
    return df


def add_date_flags(df):
    listing_after_purchase = df["ListingContractDate"] > df["PurchaseContractDate"]

    df["listing_after_close_flag"] = df["ListingContractDate"] > df["CloseDate"]
    df["purchase_after_close_flag"] = df["PurchaseContractDate"] > df["CloseDate"]
    df["negative_timeline_flag"] = (
        df["listing_after_close_flag"]
        | df["purchase_after_close_flag"]
        | listing_after_purchase
    )

    print("\nDate consistency flags")
    for flag in ["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag"]:
        print(f"  {flag}: {df[flag].sum():,} rows ({df[flag].mean() * 100:.3f}%)")
    print(f"  (listing after purchase, included in negative_timeline_flag): "
          f"{listing_after_purchase.sum():,} rows")
    return df


def add_geo_flags(df):
    lat, lon = df["Latitude"], df["Longitude"]

    missing = lat.isnull() | lon.isnull()
    zero = (lat == 0) | (lon == 0)
    positive_lon = lon > 0
    outside_box = ~(lat.between(CA_LAT_MIN, CA_LAT_MAX) & lon.between(CA_LON_MIN, CA_LON_MAX))

    df["geo_missing_flag"] = missing
    df["geo_zero_flag"] = zero & ~missing
    df["geo_positive_lon_flag"] = positive_lon & ~missing & ~zero
    df["geo_out_of_state_flag"] = outside_box & ~missing & ~zero & ~positive_lon
    geo_flags = ["geo_missing_flag", "geo_zero_flag", "geo_positive_lon_flag", "geo_out_of_state_flag"]
    df["geo_invalid_flag"] = df[geo_flags].any(axis=1)

    print("\nGeographic data quality summary")
    print(f"  Records checked: {len(df):,}")
    for flag in geo_flags + ["geo_invalid_flag"]:
        print(f"  {flag}: {df[flag].sum():,} rows ({df[flag].mean() * 100:.3f}%)")

    # Informational: could the filled-in coordinate columns repair the missing ones?
    lat_filled = pd.to_numeric(df["latfilled"], errors="coerce")
    lon_filled = pd.to_numeric(df["lonfilled"], errors="coerce")
    fixable = missing & lat_filled.notnull() & lon_filled.notnull()
    print(f"  Missing-coordinate rows that have usable latfilled/lonfilled values: {fixable.sum():,}")

    invalid = df[df["geo_invalid_flag"]]
    report_cols = ["ListingKey", "UnparsedAddress", "City", "CountyOrParish", "PostalCode",
                   "Latitude", "Longitude"] + geo_flags
    invalid[report_cols].to_csv(INVALID_COORDS_FILE, index=False)
    print(f"  Saved {len(invalid):,} invalid-coordinate records to {INVALID_COORDS_FILE.name}")
    return df


def print_missing_summary(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nRemaining missing values (left blank on purpose): {len(missing)} columns")
    summary = pd.DataFrame({"missing": missing, "pct": missing / len(df) * 100})
    print(summary.head(15).to_string())


def confirm_types(df):
    print("\nData type confirmation")
    for col in DATE_COLS:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} is not datetime"
        print(f"  {col}: {df[col].dtype}")
    for col in NUMERIC_COLS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
        print(f"  {col}: {df[col].dtype}")


def print_row_summary():
    start, end = row_log[0][1], row_log[-1][1]
    print("\nBefore/after row counts")
    for step, count in row_log:
        print(f"  {step}: {count:,}")
    print(f"  Total removed: {start - end:,} ({(start - end) / start * 100:.3f}%)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_data()
    df = convert_types(df)
    df = drop_columns(df)
    df = remove_invalid_rows(df)
    df = add_date_flags(df)
    df = add_geo_flags(df)

    # The helper coordinate columns have been checked
    df = df.drop(columns=COORD_HELPER_COLS)
    print(f"\nDropped coordinate helper columns {COORD_HELPER_COLS}")

    print_missing_summary(df)
    confirm_types(df)
    print_row_summary()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df):,} rows, {df.shape[1]} columns to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()