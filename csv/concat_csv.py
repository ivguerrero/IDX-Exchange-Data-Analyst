import glob
import re
from datetime import date
from pathlib import Path

import pandas as pd

# The script lives in the same folder as the monthly CSV files
CSV_DIR = Path(__file__).resolve().parent
START = (2024, 1)

# Most recently completed calendar month
today = date.today()
END = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)


def in_range(path):
    """True if the YYYYMM in the filename is within START..END.
    Accepts both CRMLSSold202402.csv and CRMLSSold202401_filled.csv.
    Combined output files don't end in six digits, so they are skipped."""
    m = re.search(r"(\d{4})(\d{2})(?:_filled)?\.csv$", path)
    return bool(m) and START <= (int(m.group(1)), int(m.group(2))) <= END


def build(prefix, label):
    files = sorted(f for f in glob.glob(f"{CSV_DIR}/{prefix}*.csv") if in_range(f))
    print(f"{label}: {len(files)} monthly files from {START} to {END}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        print(f"{Path(f).name}: {len(df)} rows")
        frames.append(df)

    # Row count BEFORE concatenation: sum of all monthly files
    rows_before_concat = sum(len(d) for d in frames)
    print(f"{label}: {rows_before_concat} rows before concatenation")

    combined = pd.concat(frames, ignore_index=True)

    # Row count AFTER concatenation: must match the sum above
    print(f"{label}: {len(combined)} rows after concatenation")
    assert len(combined) == rows_before_concat

    # Row count BEFORE the Residential filter (combined dataset)
    print(f"{label}: {len(combined)} rows before Residential filter")

    combined = combined[combined["PropertyType"] == "Residential"]

    # Row count AFTER the Residential filter
    print(f"{label}: {len(combined)} rows after Residential filter")

    out = CSV_DIR / f"{prefix}_combined_residential.csv"
    combined.to_csv(out, index=False)
    print(f"{label}: saved to {out}")


build("CRMLSListing", "Listings")
build("CRMLSSold", "Sold")