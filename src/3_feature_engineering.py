import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import json

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

def load_cleaned_data():
    """Load preprocessed data"""
    df = pd.read_csv(DATA_DIR / "data_cleaned.csv")
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def create_financial_ratios(df):
    """
    Create financial feature ratios.

    Why? Ratios are more interpretable and normalize for scale.
    These are standard in credit risk modeling.
    """

    print("\n" + "="*60)
    print("CREATING FINANCIAL RATIOS")
    print("="*60)

    # 1. Debt-to-Income ratio (existing feature, but we'll verify)
    if 'dti_n' in df.columns:
        df['DTI_ratio'] = df['dti_n']  # Already provided
        print("✓ DTI Ratio: Already available in data")

    # 2. Loan-to-Income ratio
    if 'revenue' in df.columns and 'loan_amnt' in df.columns:
        # Avoid division by zero
        df['loan_to_income'] = np.where(
            df['revenue'] > 0,
            df['loan_amnt'] / df['revenue'],
            0
        )
        df['loan_to_income'] = df['loan_to_income'].clip(0, 10)  # Cap at 10x income
        print("✓ Loan-to-Income Ratio: Loan Amount / Annual Revenue")

    # 3. Income brackets (categorical)
    if 'revenue' in df.columns:
        df['income_bracket'] = pd.cut(
            df['revenue'],
            bins=[0, 25000, 50000, 75000, 100000, np.inf],
            labels=['<25K', '25K-50K', '50K-75K', '75K-100K', '>100K']
        )
        print("✓ Income Bracket: Categorical grouping")

    return df

def create_temporal_features(df):
    """
    Create time-based features.

    Why? Temporal patterns in defaults (macroeconomic cycles,
    seasoning effects - loans perform differently at different ages).
    """

    print("\n" + "="*60)
    print("CREATING TEMPORAL FEATURES")
    print("="*60)

    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

        # Year issued
        df['issue_year'] = df['issue_d'].dt.year
        print("✓ Issue Year: Extract year from issue date")

        # Year-Month
        df['issue_month'] = df['issue_d'].dt.month
        print("✓ Issue Month: Extract month (seasonal patterns)")

        # Quarter
        df['issue_quarter'] = df['issue_d'].dt.quarter
        print("✓ Issue Quarter: Q1-Q4")

    return df

def create_employment_features(df):
    """
    Create employment-related features.

    Why? Employment stability is key credit risk indicator.
    """

    print("\n" + "="*60)
    print("CREATING EMPLOYMENT FEATURES")
    print("="*60)

    if 'emp_length' in df.columns:
        # Convert categorical to numeric (for regression models)
        emp_mapping = {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10,
            'n/a': 0,  # Unknown = no experience
            'Unknown': 0
        }

        df['emp_length_numeric'] = df['emp_length'].map(emp_mapping).fillna(0)
        print("✓ Employment Length (Numeric): Mapped to 0-10 scale")

        # Create employment stability groups
        df['employment_stable'] = (df['emp_length_numeric'] >= 3).astype(int)
        print("✓ Employment Stable: 1 if 3+ years, else 0")

    return df

def create_credit_score_features(df):
    """
    Create credit score-based features.

    Why? FICO score is strongest predictor of default.
    Different brackets have very different default rates.
    """

    print("\n" + "="*60)
    print("CREATING CREDIT SCORE FEATURES")
    print("="*60)

    if 'fico_n' in df.columns:
        # FICO brackets (industry standard)
        df['fico_bracket'] = pd.cut(
            df['fico_n'],
            bins=[0, 580, 670, 740, 800, 900],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        print("✓ FICO Bracket: Poor/Fair/Good/Very Good/Excellent")

        # High FICO indicator
        df['high_fico'] = (df['fico_n'] >= 740).astype(int)
        print("✓ High FICO: 1 if FICO >= 740")

    return df

def create_loan_purpose_features(df):
    """
    Create loan purpose features.

    Why? Different loan purposes have different default risks.
    Example: Debt consolidation < Car loan
    """

    print("\n" + "="*60)
    print("CREATING LOAN PURPOSE FEATURES")
    print("="*60)

    if 'purpose' in df.columns:
        # Risky vs Safe purposes
        risky_purposes = [
            'other',
            'wedding',
            'vacation',
            'small_business'
        ]

        safe_purposes = [
            'debt_consolidation',
            'home_improvement',
            'medical'
        ]

        df['risky_loan_purpose'] = df['purpose'].isin(risky_purposes).astype(int)
        df['safe_loan_purpose'] = df['purpose'].isin(safe_purposes).astype(int)

        print("✓ Risky Loan Purpose: 1 for high-risk purposes")
        print("✓ Safe Loan Purpose: 1 for safe purposes")

        # Purpose categories (one-hot encoding will be done later)
        print(f"✓ Unique purposes: {df['purpose'].nunique()}")

    return df

def create_home_ownership_features(df):
    """
    Home ownership features.

    Why? Homeownership correlates with stability and default risk.
    """

    print("\n" + "="*60)
    print("CREATING HOME OWNERSHIP FEATURES")
    print("="*60)

    if 'home_ownership_n' in df.columns:
        # Home owner indicator
        df['is_homeowner'] = (df['home_ownership_n'] == 'OWN').astype(int)
        df['has_mortgage'] = (df['home_ownership_n'] == 'MORTGAGE').astype(int)
        df['is_renter'] = (df['home_ownership_n'] == 'RENT').astype(int)

        print("✓ Is Homeowner: 1 if OWN")
        print("✓ Has Mortgage: 1 if MORTGAGE")
        print("✓ Is Renter: 1 if RENT")

    return df

def create_interaction_features(df):
    """
    Create interaction features.

    Why? Combinations of features can capture complex relationships.
    Example: High debt AND low income = very risky
    """

    print("\n" + "="*60)
    print("CREATING INTERACTION FEATURES")
    print("="*60)

    # High debt + Low income
    if 'DTI_ratio' in df.columns and 'income_bracket' in df.columns:
        high_dti = df['DTI_ratio'] > df['DTI_ratio'].quantile(0.75)
        low_income = df['income_bracket'].isin(['<25K', '25K-50K'])
        df['high_debt_low_income'] = (high_dti & low_income).astype(int)
        print("✓ High Debt + Low Income: Risky combination")

    # High loan amount + Low FICO
    if 'loan_amnt' in df.columns and 'fico_n' in df.columns:
        high_loan = df['loan_amnt'] > df['loan_amnt'].quantile(0.75)
        low_fico = df['fico_n'] < df['fico_n'].quantile(0.25)
        df['high_loan_low_fico'] = (high_loan & low_fico).astype(int)
        print("✓ High Loan + Low FICO: High risk")

    return df

def select_features_for_modeling(df):
    """
    Select features for modeling.

    Strategy:
    - Numeric: All ratio and score features
    - Categorical: One-hot encode with low cardinality
    - Drop: ID, dates (already extracted), redundant features
    """

    print("\n" + "="*60)
    print("FEATURE SELECTION FOR MODELING")
    print("="*60)

    # Numeric features (keep for regression)
    numeric_features = [
        'revenue', 'loan_amnt', 'fico_n', 'dti_n', 'emp_length_numeric',
        'loan_to_income', 'DTI_ratio', 'issue_year', 'issue_month', 'issue_quarter'
    ]

    # Binary/Categorical features
    binary_features = [
        'employment_stable', 'high_fico', 'is_homeowner', 'has_mortgage',
        'is_renter', 'risky_loan_purpose', 'safe_loan_purpose',
        'high_debt_low_income', 'high_loan_low_fico'
    ]

    # Categorical features to one-hot encode
    categorical_features = [
        'purpose', 'home_ownership_n', 'income_bracket', 'fico_bracket'
    ]

    # Keep only features that exist
    numeric_features = [f for f in numeric_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    selected_features = numeric_features + binary_features + categorical_features

    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"Binary features ({len(binary_features)}): {binary_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"\nTotal selected features: {len(selected_features)}")

    # Keep target + selected features
    df_selected = df[selected_features + ['Default']].copy()

    return df_selected, numeric_features, binary_features, categorical_features

def handle_categorical_encoding(df, categorical_features):
    """
    One-hot encode categorical features.

    Why one-hot?
    - Tree models (XGBoost): Can use directly
    - Logistic Regression: Requires numeric
    - Interpretable: Shows effect of each category
    """

    print("\n" + "="*60)
    print("CATEGORICAL ENCODING (ONE-HOT)")
    print("="*60)

    # One-hot encode with drop_first=True to avoid multicollinearity
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True,
        prefix=categorical_features
    )

    print(f"\n✓ Before encoding: {len(df.columns)} columns")
    print(f"✓ After encoding: {len(df_encoded.columns)} columns")
    print(f"✓ New categorical columns: {len(df_encoded.columns) - len(df.columns)}")

    return df_encoded

def create_feature_scaling_pipeline(df, numeric_features):
    """
    Create scaling pipeline for numeric features.

    Why StandardScaler?
    - Logistic Regression: Requires normalization
    - Improves convergence
    - Fair coefficient comparison
    - Tree models don't need it, but doesn't hurt
    """

    print("\n" + "="*60)
    print("CREATING SCALING PIPELINE")
    print("="*60)

    scaler = StandardScaler()

    # Fit on numeric features
    scaler.fit(df[numeric_features])

    # Save for later use in production
    import pickle
    with open(MODELS_DIR / "feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✓ StandardScaler fitted on {len(numeric_features)} numeric features")
    print(f"✓ Saved to: {MODELS_DIR / 'feature_scaler.pkl'}")

    # Scale features
    df[numeric_features] = scaler.transform(df[numeric_features])

    print(f"\n✓ Scaled data statistics:")
    print(f"  Mean (should be ~0): {df[numeric_features].mean().mean():.6f}")
    print(f"  Std (should be ~1): {df[numeric_features].std().mean():.6f}")

    return df, scaler

def save_feature_info(df, numeric_features, categorical_features, binary_features):
    """
    Save feature metadata for later use.
    """

    feature_info = {
        'all_features': df.columns.tolist(),
        'numeric_features': numeric_features,
        'binary_features': binary_features,
        'categorical_features': categorical_features,
        'target': 'Default',
        'n_features': len(df.columns) - 1,  # Exclude target
        'n_samples': len(df)
    }

    with open(MODELS_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\n✓ Feature info saved to: {MODELS_DIR / 'feature_names.json'}")

    return feature_info

def main():
    """Execute feature engineering pipeline"""

    print("\n" + "="*70)
    print("PHASE 5: FEATURE ENGINEERING & TRANSFORMATION")
    print("="*70)

    # Load
    df = load_cleaned_data()

    # Create features
    df = create_financial_ratios(df)
    df = create_temporal_features(df)
    df = create_employment_features(df)
    df = create_credit_score_features(df)
    df = create_loan_purpose_features(df)
    df = create_home_ownership_features(df)
    df = create_interaction_features(df)

    # Select and encode
    df, numeric_features, binary_features, categorical_features = select_features_for_modeling(df)

    df = handle_categorical_encoding(df, categorical_features)

    df, scaler = create_feature_scaling_pipeline(df, numeric_features)

    # Save feature info
    feature_info = save_feature_info(df, numeric_features, categorical_features, binary_features)

    # Save engineered dataset
    df.to_csv(DATA_DIR / "features_engineered.csv", index=False)
    print(f"\n✓ Engineered features saved to: {DATA_DIR / 'features_engineered.csv'}")

    print("\n" + "="*70)
    print("✓ PHASE 5 COMPLETE")
    print("="*70)
    print(f"\nFinal dataset:")
    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Default rate: {df['Default'].mean()*100:.2f}%")
    print("\nNext: Run `python src/4_model_training.py`")

if __name__ == "__main__":
    main()