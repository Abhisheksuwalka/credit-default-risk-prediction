import pandas as pd
import numpy as np
import shap
import pickle
from pathlib import Path

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")

def load_model_and_data():
    """Load trained XGBoost and test data"""

    # Load model
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    # Load test data
    df = pd.read_csv(DATA_DIR / "features_engineered.csv")

    # Use last 1000 samples for SHAP (computational cost)
    X = df.drop('Default', axis=1).tail(1000)
    y = df['Default'].tail(1000)

    return xgb_model, X, y

def explain_with_shap(model, X):
    """
    Generate SHAP values for model explanation.

    SHAP (SHapley Additive exPlanations):
    - Game theory approach
    - Measures feature contribution to each prediction
    - Global + local explanations
    """

    print("\n" + "="*60)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*60)

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, get class 1 (default) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"✓ SHAP values computed for {X.shape[0]} samples")
    print(f"✓ Shape: {shap_values.shape}")

    return explainer, shap_values

def save_shap_results(explainer, shap_values, X):
    """Save SHAP results for frontend visualization"""

    # Mean absolute SHAP values (feature importance)
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Save
    importance_df.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)

    return importance_df

def main():
    """Execute SHAP analysis"""

    print("\n" + "="*70)
    print("PHASE 7: SHAP MODEL EXPLAINABILITY")
    print("="*70)

    model, X, y = load_model_and_data()
    explainer, shap_values = explain_with_shap(model, X)
    importance_df = save_shap_results(explainer, shap_values, X)

    print("\n✓ SHAP analysis complete!")
    print("Feature importance visualization will be in frontend")

if __name__ == "__main__":
    main()


# ======================================================================
# PHASE 7: SHAP MODEL EXPLAINABILITY
# ======================================================================

# ============================================================
# GENERATING SHAP EXPLANATIONS
# ============================================================
# [14:29:35] WARNING: /Users/runner/work/xgboost/xgboost/src/c_api/c_api.cc:1240: Saving into deprecated binary model format, please consider using `json` or `ubj`. Model format will default to JSON in XGBoost 2.2 if not specified.
# ✓ SHAP values computed for 1000 samples
# ✓ Shape: (1000, 43)

# Top 10 Most Important Features:
#             Feature  Importance
#              fico_n    0.297678
#      loan_to_income    0.244071
#          issue_year    0.242841
#               dti_n    0.169224
#           loan_amnt    0.089780
#        has_mortgage    0.085552
# purpose_credit_card    0.081935
#           is_renter    0.079034
#  emp_length_numeric    0.055677
#             revenue    0.030952

# ✓ SHAP analysis complete!
# Feature importance visualization will be in frontend