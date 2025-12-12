import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("../data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

def load_raw_data():
    """Load raw dataset"""
    df = pd.read_csv(DATA_DIR / "raw" / "lending_club_loans.csv", low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def analyze_missing_values(df):
    """
    Comprehensive missing value analysis.

    Decision Logic:
    - >50% missing: DROP (not useful)
    - 5-50% missing: Impute intelligently
    - <5% missing: Drop rows or impute
    """

    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)

    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
    }).sort_values('Missing_Percentage', ascending=False)

    missing = missing[missing['Missing_Count'] > 0]

    if len(missing) > 0:
        print("\nColumns with missing values:")
        print(missing.to_string(index=False))
    else:
        print("\n✓ No missing values detected!")

    # Decision: Drop features with >50% missing
    high_missing = missing[missing['Missing_Percentage'] > 50]['Column'].tolist()
    if high_missing:
        print(f"\n⚠️ Dropping columns with >50% missing: {high_missing}")
        df = df.drop(columns=high_missing)

    # Decision: For numeric features with <5% missing, drop rows
    low_missing = missing[missing['Missing_Percentage'] < 5]['Column'].tolist()
    numeric_missing = [col for col in low_missing if df[col].dtype != 'object']

    if numeric_missing:
        print(f"\n✓ Dropping {len(df)} rows with <5% missing numeric values")
        df = df.dropna(subset=numeric_missing)

    # Decision: For categorical features, impute with 'Unknown'
    cat_missing = [col for col in low_missing if df[col].dtype == 'object']
    if cat_missing:
        print(f"\n✓ Imputing categorical features: {cat_missing}")
        df[cat_missing] = df[cat_missing].fillna('Unknown')

    return df

def analyze_data_types(df):
    """
    Identify and convert data types correctly.

    Why? Ensures memory efficiency and correct operations.
    """

    print("\n" + "="*60)
    print("DATA TYPE ANALYSIS")
    print("="*60)

    # Numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:5]}...")

    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Convert high-cardinality text to category (memory optimization)
    for col in categorical_cols:
        if df[col].nunique() < 50:
            df[col] = df[col].astype('category')
            print(f"  ✓ Converted {col} to category ({df[col].nunique()} unique values)")

    return df

def analyze_outliers(df, numeric_cols):
    """
    Detect and handle outliers using IQR method.

    Why IQR?
    - Robust to extreme values
    - Works for non-normal distributions
    - Industry standard in finance
    """

    print("\n" + "="*60)
    print("OUTLIER DETECTION (IQR Method)")
    print("="*60)

    outlier_summary = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if len(outliers) > 0:
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

    if outlier_summary:
        print("\nColumns with outliers:")
        for col, stats in sorted(outlier_summary.items(),
                                 key=lambda x: x[1]['percentage'],
                                 reverse=True)[:10]:
            print(f"  {col}: {stats['count']:,} outliers ({stats['percentage']:.2f}%)")

    # Decision: Cap outliers at 1st/99th percentile (not drop)
    # Why? Outliers in credit risk are informative (high income, high debt are real)
    print("\n✓ Capping outliers at 1st/99th percentile (not removing)")

    for col in numeric_cols:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p1, upper=p99)

    return df

def analyze_target_distribution(df):
    """
    Analyze default rate and class imbalance.

    Decision: Use stratified sampling to maintain class balance.
    """

    print("\n" + "="*60)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*60)

    default_counts = df['Default'].value_counts()
    default_pct = df['Default'].value_counts(normalize=True) * 100

    print("\nDefault vs Fully Paid:")
    for status, count in default_counts.items():
        pct = default_pct[status]
        status_name = "Fully Paid" if status == 0 else "Default"
        print(f"  {status_name}: {count:,} ({pct:.2f}%)")

    print(f"\nClass Imbalance Ratio: {default_counts[0] / default_counts[1]:.2f}:1")
    print("Note: Will use stratified sampling & class weights during modeling")

    return df

def statistical_summary(df, numeric_cols):
    """
    Generate comprehensive statistical summary.
    """

    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (Numeric Features)")
    print("="*60)

    stats = df[numeric_cols].describe().T
    stats['skewness'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()

    print("\nTop 5 features by standard deviation:")
    print(stats.sort_values('std', ascending=False)[['mean', 'std', 'min', 'max']].head())

    # Save for later reference
    stats.to_csv(PROCESSED_DIR / "statistical_summary.csv")
    print("\n✓ Saved to: data/processed/statistical_summary.csv")

    return stats

def create_preprocessed_dataset(df, numeric_cols):
    """
    Create cleaned, standardized dataset.
    """

    print("\n" + "="*60)
    print("CREATING PREPROCESSED DATASET")
    print("="*60)

    df_processed = df.copy()

    # Remove rows where target is missing
    df_processed = df_processed[df_processed['Default'].notna()]

    # Ensure no infinite values
    for col in numeric_cols:
        df_processed = df_processed[~np.isinf(df_processed[col])]

    print(f"\n✓ Final dataset shape: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
    print(f"✓ Default rate maintained: {df_processed['Default'].mean()*100:.2f}%")

    # Save
    df_processed.to_csv(PROCESSED_DIR / "data_cleaned.csv", index=False)
    print(f"\n✓ Saved to: {PROCESSED_DIR / 'data_cleaned.csv'}")

    return df_processed

def main():
    """Execute full EDA pipeline"""

    print("\n" + "="*70)
    print("PHASE 4: EXPLORATORY DATA ANALYSIS & DATA CLEANING")
    print("="*70)

    # Load data
    df = load_raw_data()

    # Cleaning pipeline
    df = analyze_missing_values(df)
    df = analyze_data_types(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    df = analyze_outliers(df, numeric_cols)
    df = analyze_target_distribution(df)

    # Analysis
    stats = statistical_summary(df, numeric_cols)

    # Save
    df = create_preprocessed_dataset(df, numeric_cols)

    print("\n" + "="*70)
    print("✓ PHASE 4 COMPLETE")
    print("="*70)
    print("\nNext: Run `python src/3_feature_engineering.py`")

if __name__ == "__main__":
    main()