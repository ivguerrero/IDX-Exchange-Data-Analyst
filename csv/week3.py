from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOLD_FILE = BASE_DIR / "CRMLSSold_combined_residential.csv"
LISTING_FILE = BASE_DIR / "CRMLSListing_combined_residential.csv"

sold = pd.read_csv(SOLD_FILE, low_memory = False)
listings = pd.read_csv(LISTING_FILE, low_memory = False)

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

# Step 2 – Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (mortgage.groupby('year_month')['rate_30yr_fixed'].mean().reset_index())

# Step 3 – Create a matching year_month key on the MLS datasets
# Sold dataset — key off CloseDate
sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')
# Listings dataset — key off ListingContractDate
listings['year_month'] = pd.to_datetime(listings['ListingContractDate']).dt.to_period('M') 

# Step 4 – Merge
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')

#save to csv
SOLD_OUT = BASE_DIR / "CRMLSSold_enriched.csv"
LISTING_OUT = BASE_DIR / "CRMLSListing_enriched.csv"

sold_with_rates.to_csv(SOLD_OUT, index=False)
listings_with_rates.to_csv(LISTING_OUT, index=False)

# Step 5 – Validate the merge
assert len(sold_with_rates) == len(sold), "Sold row count changed after merge"
assert len(listings_with_rates) == len(listings), "Listings row count changed after merge"

sold_nulls = sold_with_rates['rate_30yr_fixed'].isnull().sum()
listing_nulls = listings_with_rates['rate_30yr_fixed'].isnull().sum()
print(f"Null rates: sold {sold_nulls}, listings {listing_nulls}")

assert sold_nulls == 0, "Sold has rows with no mortgage rate"
assert listing_nulls == 0, "Listings has rows with no mortgage rate"

# Preview
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())