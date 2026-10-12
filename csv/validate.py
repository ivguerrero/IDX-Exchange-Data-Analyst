import pandas as pd

sold = pd.read_csv("/Users/ivanguerrero/IDXExchange/csv/CRMLSSold_combined_residential.csv", low_memory = False)
missing = sold.isnull().sum()

for col, count in missing.items():
    print(f"{col} : {count} missing. {count / len(sold) * 100:.1f}%")
    if ((count / len(sold) * 100) > 90 ):
        print("*****************FLAGGED********************")