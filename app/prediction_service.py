import pickle
import json
import numpy as np
from pathlib import Path

class PredictionService:
    """Service for making predictions"""
    
    def __init__(self, models_dir="models"):
        """Initialize with trained models"""
        self.models_dir = Path(models_dir)
        
        # Load models
        print('LOADING logistic_regression_model.pkl')
        with open(self.models_dir / "logistic_regression_model.pkl", 'rb') as f:
            self.lr_model = pickle.load(f)
            
        
        with open(self.models_dir / "feature_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(self.models_dir / "feature_names.json", 'r') as f:
            self.feature_config = json.load(f)
    
    def predict_draft(self, revenue, loan_amnt, fico_n, dti_n, emp_length_numeric):
        """
        Make prediction for loan application
        
        Returns:
            dict: Prediction with probabilities and risk category
        """
        
        # # Create feature array
        # features = np.array([
        #     revenue,
        #     loan_amnt,
        #     fico_n,
        #     dti_n,
        #     emp_length_numeric
        # ]).reshape(1, -1)
        
        # # Scale features
        # features_scaled = self.scaler.transform(features)
        
        # # Get probability
        # pd_probability = float(self.lr_model.predict_proba(features_scaled)[0, 1])
        
        # Derive missing numeric features to match training data (10 total)
        loan_to_income = loan_amnt / revenue
        dti_percentage = dti_n * 100  # Convert input ratio to % for dti_n feature (matches training data)
        dti_ratio = dti_n  # Direct ratio for DTI_ratio feature
        issue_year = 2025.0
        issue_month = 12.0
        issue_quarter = 4.0
        
        # Create full numeric feature array (order matches numeric_features in feature_names.json)
        features = np.array([
            revenue,
            loan_amnt,
            fico_n,
            dti_percentage,  # dti_n (as %)
            emp_length_numeric,
            loan_to_income,
            dti_ratio,  # DTI_ratio
            issue_year,
            issue_month,
            issue_quarter
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability
        pd_probability = float(self.lr_model.predict_proba(features_scaled)[0, 1])
        
        # Determine risk category
        if pd_probability < 0.10:
            risk_category = "Low Risk"
            recommendation = "✓ APPROVE"
            confidence = 0.95
        elif pd_probability < 0.25:
            risk_category = "Medium Risk"
            recommendation = "⚠ REVIEW"
            confidence = 0.80
        else:
            risk_category = "High Risk"
            recommendation = "✗ DECLINE"
            confidence = 0.90
        
        # Identify risk factors
        risk_factors = []
        if dti_n > 0.4:
            risk_factors.append("High debt-to-income ratio")
        if fico_n < 650:
            risk_factors.append("Low credit score")
        if loan_amnt > revenue * 0.5:
            risk_factors.append("High loan-to-income ratio")
        if emp_length_numeric < 2:
            risk_factors.append("Short employment history")
        
        return {
            "probability_of_default": pd_probability,
            "probability_percentage": f"{pd_probability*100:.2f}%",
            "risk_category": risk_category,
            "confidence_score": confidence,
            "recommendation": recommendation,
            "top_risk_factors": risk_factors
        }
        




    def predict(self, revenue, loan_amnt, fico_n, dti_n, emp_length_numeric):
        """
        Make prediction for loan application
        
        Returns:
            dict: Prediction with probabilities and risk category
        """
        
        # Load all input features (exclude target 'Default')
        all_features = [f for f in self.feature_config['all_features'] if f != 'Default']
        
        # Initialize full feature dict with zeros (use float for consistency with scaling)
        full_features = {col: 0.0 for col in all_features}
        
        # Set raw numeric features (will scale later)
        full_features['revenue'] = float(revenue)
        full_features['loan_amnt'] = float(loan_amnt)
        full_features['fico_n'] = float(fico_n)
        full_features['dti_n'] = dti_n * 100  # Convert to % to match training data
        full_features['emp_length_numeric'] = float(emp_length_numeric)
        full_features['loan_to_income'] = loan_amnt / revenue
        full_features['DTI_ratio'] = dti_n  # Keep as ratio (0-1)
        full_features['issue_year'] = 2025.0
        full_features['issue_month'] = 12.0
        full_features['issue_quarter'] = 4.0
        
        # Set binary features (derived from inputs/defaults)
        full_features['employment_stable'] = 1.0 if emp_length_numeric >= 3 else 0.0
        full_features['high_fico'] = 1.0 if fico_n >= 700 else 0.0
        full_features['is_homeowner'] = 0.0  # Default: RENT
        full_features['has_mortgage'] = 0.0  # Default: not MORTGAGE
        full_features['is_renter'] = 1.0  # Default: RENT
        full_features['risky_loan_purpose'] = 0.0  # Default purpose low-risk
        full_features['safe_loan_purpose'] = 1.0  # Default purpose safe
        full_features['high_debt_low_income'] = 1.0 if dti_n > 0.36 and revenue < 50000 else 0.0
        full_features['high_loan_low_fico'] = 1.0 if loan_amnt > 20000 and fico_n < 660 else 0.0
        
        # Set one-hot categorical features
        # Purpose: default to 'debt_consolidation' (common, medium-low risk per LendingClub data)
        full_features['purpose_debt_consolidation'] = 1.0
        
        # Home ownership: default to 'RENT' (common for applicants)
        full_features['home_ownership_n_RENT'] = 1.0
        
        # Income bracket (bins from engineering: drop <25K)
        if 25000 <= revenue < 50000:
            full_features['income_bracket_25K-50K'] = 1.0
        elif 50000 <= revenue < 75000:
            full_features['income_bracket_50K-75K'] = 1.0
        elif 75000 <= revenue < 100000:
            full_features['income_bracket_75K-100K'] = 1.0
        elif revenue >= 100000:
            full_features['income_bracket_>100K'] = 1.0
        # else: all 0 = <25K (reference)
        
        # FICO bracket (standard LendingClub/FICO: drop 'Fair')
        if 670 <= fico_n < 740:
            full_features['fico_bracket_Good'] = 1.0
        elif 740 <= fico_n < 800:
            full_features['fico_bracket_Very Good'] = 1.0
        elif fico_n >= 800:
            full_features['fico_bracket_Excellent'] = 1.0
        # else: all 0 = Fair (<670, reference)
        
        # Create DataFrame
        df = pd.DataFrame([full_features])
        
        # Scale only numeric features (uses column names to match training, suppresses warning)
        numeric_features = self.feature_config['numeric_features']
        df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        # Predict on full feature set
        pd_probability = float(self.lr_model.predict_proba(df[all_features])[0, 1])
        
        # Determine risk category
        if pd_probability < 0.10:
            risk_category = "Low Risk"
            recommendation = "✓ APPROVE"
            confidence = 0.95
        elif pd_probability < 0.25:
            risk_category = "Medium Risk"
            recommendation = "⚠ REVIEW"
            confidence = 0.80
        else:
            risk_category = "High Risk"
            recommendation = "✗ DECLINE"
            confidence = 0.90
        
        # Identify risk factors
        risk_factors = []
        if dti_n > 0.4:
            risk_factors.append("High debt-to-income ratio")
        if fico_n < 650:
            risk_factors.append("Low credit score")
        if loan_amnt > revenue * 0.5:
            risk_factors.append("High loan-to-income ratio")
        if emp_length_numeric < 2:
            risk_factors.append("Short employment history")
        
        return {
            "probability_of_default": pd_probability,
            "probability_percentage": f"{pd_probability*100:.2f}%",
            "risk_category": risk_category,
            "confidence_score": confidence,
            "recommendation": recommendation,
            "top_risk_factors": risk_factors
        }