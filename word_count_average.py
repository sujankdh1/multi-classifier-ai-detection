import pandas as pd
import numpy as np

# Read the CSV file
print("Loading CSV file...")
df = pd.read_csv('LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv')

# Check column names
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check if required columns exist
if 'WC' not in df.columns:
    print("Error: 'WC' column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Check for flagged column (could be is_ai_flagged or is_flagged)
flagged_col = None
if 'is_ai_flagged' in df.columns:
    flagged_col = 'is_ai_flagged'
elif 'is_flagged' in df.columns:
    flagged_col = 'is_flagged'
else:
    print("Warning: No 'is_ai_flagged' or 'is_flagged' column found. Calculating overall average only.")
    print(f"Available columns: {df.columns.tolist()}")

# Convert WC to numeric, handling any non-numeric values
df['WC'] = pd.to_numeric(df['WC'], errors='coerce')

# Remove rows with NaN word counts
df_clean = df.dropna(subset=['WC'])

print(f"\nRows with valid word counts: {len(df_clean)}")

# Calculate overall average
overall_avg = df_clean['WC'].mean()
overall_median = df_clean['WC'].median()
overall_std = df_clean['WC'].std()

print("\n" + "="*60)
print("OVERALL WORD COUNT STATISTICS")
print("="*60)
print(f"Average word count: {overall_avg:.2f}")
print(f"Median word count: {overall_median:.2f}")
print(f"Standard deviation: {overall_std:.2f}")
print(f"Min word count: {df_clean['WC'].min():.0f}")
print(f"Max word count: {df_clean['WC'].max():.0f}")

# If flagged column exists, calculate averages by flagged status
if flagged_col:
    # Convert flagged column to boolean if needed
    if df_clean[flagged_col].dtype == 'object':
        # Try to convert string values to boolean
        df_clean[flagged_col] = df_clean[flagged_col].astype(str).str.lower().str.strip()
        df_clean[flagged_col] = df_clean[flagged_col].isin(['true', '1', 'yes', 'y', 'flagged'])
    
    # Group by flagged status
    grouped = df_clean.groupby(flagged_col)['WC']
    
    print("\n" + "="*60)
    print("WORD COUNT STATISTICS BY FLAGGED STATUS")
    print("="*60)
    
    for flag_status in sorted(df_clean[flagged_col].unique()):
        flag_data = df_clean[df_clean[flagged_col] == flag_status]
        count = len(flag_data)
        avg = flag_data['WC'].mean()
        median = flag_data['WC'].median()
        std = flag_data['WC'].std()
        
        status_name = "Flagged" if flag_status else "Not Flagged"
        print(f"\n{status_name} (value: {flag_status}):")
        print(f"  Count: {count:,}")
        print(f"  Average word count: {avg:.2f}")
        print(f"  Median word count: {median:.2f}")
        print(f"  Standard deviation: {std:.2f}")
        print(f"  Min: {flag_data['WC'].min():.0f}")
        print(f"  Max: {flag_data['WC'].max():.0f}")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
