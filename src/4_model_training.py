"""
Model Training Module for Credit Default Risk Prediction
=========================================================

Purpose:
    - Train multiple ML models (Logistic Regression, XGBoost)
    - Perform hyperparameter optimization
    - Comprehensive model evaluation with industry metrics
    - Save trained models and metadata

Key Metrics:
    - AUC-ROC: Overall discrimination ability
    - Gini Coefficient: (2 × AUC) - 1 (credit industry standard)
    - KS Statistic: Max separation between default/non-default
    - Precision/Recall: Balance for imbalanced data

Author: Credit Risk Analytics Team
Date: December 2024
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, 
    cross_validate, 
    StratifiedKFold,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

# Model training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 25% of remaining 80% = 20% overall
N_JOBS = -1  # Use all available cores
CV_FOLDS = 5

# Evaluation thresholds
THRESHOLD_LOW_RISK = 0.10
THRESHOLD_HIGH_RISK = 0.25

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def save_json(data: dict, filepath: Path):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✓ Saved: {filepath}")


# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================

def load_processed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature-engineered dataset.
    
    Returns:
        Tuple of (X: features DataFrame, y: target Series)
    """
    
    print_section("LOADING PROCESSED DATA")
    
    filepath = DATA_DIR / "features_engineered.csv"
    
    if not filepath.exists():
        print(f"✗ Error: {filepath} not found")
        print("Please run src/3_feature_engineering.py first")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    
    # Separate features and target
    if 'Default' not in df.columns:
        print("✗ Error: Target column 'Default' not found")
        sys.exit(1)
    
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    print(f"✓ Loaded data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Default rate: {y.mean()*100:.2f}%")
    print(f"  Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return X, y


def create_stratified_split(
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create stratified train/validation/test splits.
    
    Strategy:
    - 60% train
    - 20% validation  
    - 20% test
    - Stratification maintains class balance
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    print_section("CREATING DATA SPLITS")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    # Verify splits
    total_samples = len(X)
    
    print(f"Total samples: {total_samples:,}")
    print(f"\nTrain set:")
    print(f"  Samples: {len(X_train):,} ({len(X_train)/total_samples*100:.1f}%)")
    print(f"  Default rate: {y_train.mean()*100:.2f}%")
    
    print(f"\nValidation set:")
    print(f"  Samples: {len(X_val):,} ({len(X_val)/total_samples*100:.1f}%)")
    print(f"  Default rate: {y_val.mean()*100:.2f}%")
    
    print(f"\nTest set:")
    print(f"  Samples: {len(X_test):,} ({len(X_test)/total_samples*100:.1f}%)")
    print(f"  Default rate: {y_test.mean()*100:.2f}%")
    
    # Save split info
    split_info = {
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_default_rate': float(y_train.mean()),
        'val_default_rate': float(y_val.mean()),
        'test_default_rate': float(y_test.mean()),
    }
    
    save_json(split_info, DATA_DIR / "data_splits.json")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_grid_search: bool = True
) -> Tuple[LogisticRegression, Dict]:
    """
    Train Logistic Regression with hyperparameter tuning.
    
    Why Logistic Regression?
    - Industry standard in credit risk (Basel, IFRS 9)
    - Fully interpretable coefficients
    - Regulatory-friendly (transparent decisions)
    - Fast training and inference
    - Probabilistic outputs
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        use_grid_search: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained model, training metrics dict)
    """
    
    print_section("TRAINING: LOGISTIC REGRESSION")
    
    if use_grid_search:
        print("⏳ Performing Grid Search for hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
            'penalty': ['l2'],  # L2 regularization
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000],
            'class_weight': ['balanced']
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=N_JOBS,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV AUC: {grid_search.best_score_:.4f}")
        
    else:
        # Use default parameters with class weighting
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        
        model.fit(X_train, y_train)
        print("✓ Model trained with default parameters")
    
    # Validation predictions
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    print(f"\nValidation Performance:")
    print(f"  AUC-ROC: {val_auc:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    
    training_info = {
        'model_type': 'Logistic Regression',
        'parameters': model.get_params(),
        'val_auc': val_auc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'training_samples': len(X_train),
        'training_time': datetime.now().isoformat()
    }
    
    return model, training_info


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_grid_search: bool = False
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost with early stopping and hyperparameter tuning.
    
    Why XGBoost?
    - State-of-the-art gradient boosting
    - Captures non-linear relationships automatically
    - Handles feature interactions
    - Built-in regularization
    - Excellent on tabular data
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        use_grid_search: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained model, training metrics dict)
    """
    
    print_section("TRAINING: XGBOOST")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    if use_grid_search:
        print("\n⏳ Performing Grid Search (this may take a while)...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 6, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [scale_pos_weight]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                eval_metric='logloss'
            ),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=1,  # XGBoost uses parallel internally
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV AUC: {grid_search.best_score_:.4f}")
        
    else:
        # Use optimized default parameters
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print(f"✓ Model trained with early stopping")
        print(f"✓ Best iteration: {model.best_iteration}")
    
    # Validation predictions
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    print(f"\nValidation Performance:")
    print(f"  AUC-ROC: {val_auc:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    
    training_info = {
        'model_type': 'XGBoost',
        'parameters': model.get_params(),
        'val_auc': val_auc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'training_samples': len(X_train),
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else None,
        'training_time': datetime.now().isoformat()
    }
    
    return model, training_info


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    KS = max(TPR - FPR) across all thresholds
    
    Industry interpretation:
    - KS > 40%: Excellent model
    - 30-40%: Good model
    - 20-30%: Acceptable
    - < 20%: Poor
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        KS statistic (0-1)
    """
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = max(tpr - fpr)
    return ks


def evaluate_model_comprehensive(
    model,
    model_name: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Comprehensive model evaluation with industry metrics.
    
    Metrics calculated:
    - AUC-ROC: Overall discrimination ability
    - Gini: (2 × AUC) - 1 (credit industry standard)
    - KS Statistic: Regulatory requirement
    - Precision/Recall: Class balance metrics
    - Confusion Matrix: Error analysis
    
    Args:
        model: Trained model
        model_name: Model identifier
        X_val, y_val: Validation data
        X_test, y_test: Test data
    
    Returns:
        Dictionary with all metrics
    """
    
    print_section(f"COMPREHENSIVE EVALUATION: {model_name}")
    
    # Get predictions for both sets
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # -------------------------------------------------------------------------
    # VALIDATION SET METRICS
    # -------------------------------------------------------------------------
    
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_gini = 2 * val_auc - 1
    val_ks = calculate_ks_statistic(y_val, y_val_pred_proba)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    # -------------------------------------------------------------------------
    # TEST SET METRICS
    # -------------------------------------------------------------------------
    
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_gini = 2 * test_auc - 1
    test_ks = calculate_ks_statistic(y_test, y_test_pred_proba)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # -------------------------------------------------------------------------
    # CONFUSION MATRIX (Test Set)
    # -------------------------------------------------------------------------
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    
    print("VALIDATION SET METRICS:")
    print(f"  AUC-ROC:    {val_auc:.4f}")
    print(f"  Gini:       {val_gini:.4f}")
    print(f"  KS Stat:    {val_ks:.4f}")
    print(f"  Accuracy:   {val_accuracy:.4f}")
    print(f"  Precision:  {val_precision:.4f}")
    print(f"  Recall:     {val_recall:.4f}")
    print(f"  F1 Score:   {val_f1:.4f}")
    
    print("\nTEST SET METRICS:")
    print(f"  AUC-ROC:    {test_auc:.4f}")
    print(f"  Gini:       {test_gini:.4f}")
    print(f"  KS Stat:    {test_ks:.4f}")
    print(f"  Accuracy:   {test_accuracy:.4f}")
    print(f"  Precision:  {test_precision:.4f}")
    print(f"  Recall:     {test_recall:.4f}")
    print(f"  F1 Score:   {test_f1:.4f}")
    
    print("\nCONFUSION MATRIX (Test Set):")
    print(f"  True Negatives:   {tn:,}")
    print(f"  False Positives:  {fp:,}")
    print(f"  False Negatives:  {fn:,}")
    print(f"  True Positives:   {tp:,}")
    
    # -------------------------------------------------------------------------
    # BUSINESS INTERPRETATION
    # -------------------------------------------------------------------------
    
    print("\nBUSINESS INTERPRETATION:")
    
    # Gini interpretation
    if test_gini > 0.5:
        gini_interp = "Excellent - Strong discrimination"
    elif test_gini > 0.4:
        gini_interp = "Good - Acceptable discrimination"
    elif test_gini > 0.3:
        gini_interp = "Fair - Minimum acceptable"
    else:
        gini_interp = "Poor - Needs improvement"
    print(f"  Gini ({test_gini:.3f}): {gini_interp}")
    
    # KS interpretation
    if test_ks > 0.4:
        ks_interp = "Excellent separation"
    elif test_ks > 0.3:
        ks_interp = "Good separation"
    elif test_ks > 0.2:
        ks_interp = "Acceptable separation"
    else:
        ks_interp = "Poor separation"
    print(f"  KS ({test_ks:.3f}): {ks_interp}")
    
    # -------------------------------------------------------------------------
    # COMPILE METRICS DICTIONARY
    # -------------------------------------------------------------------------
    
    metrics = {
        'model_name': model_name,
        
        # Validation metrics
        'val_auc': float(val_auc),
        'val_gini': float(val_gini),
        'val_ks': float(val_ks),
        'val_accuracy': float(val_accuracy),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        
        # Test metrics
        'test_auc': float(test_auc),
        'test_gini': float(test_gini),
        'test_ks': float(test_ks),
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        
        # Confusion matrix
        'test_tn': int(tn),
        'test_fp': int(fp),
        'test_fn': int(fn),
        'test_tp': int(tp),
        
        # Interpretations
        'gini_interpretation': gini_interp,
        'ks_interpretation': ks_interp,
    }
    
    return metrics


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_models(lr_model, xgb_model, lr_metrics: Dict, xgb_metrics: Dict):
    """
    Save trained models and metadata.
    
    Args:
        lr_model: Trained Logistic Regression
        xgb_model: Trained XGBoost
        lr_metrics: LR evaluation metrics
        xgb_metrics: XGB evaluation metrics
    """
    
    print_section("SAVING MODELS")
    
    # Save Logistic Regression
    lr_path = MODELS_DIR / "logistic_regression_model.pkl"
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✓ Saved: {lr_path}")
    
    # Save XGBoost (use native format)
    xgb_path = MODELS_DIR / "xgboost_model.json"
    xgb_model.save_model(str(xgb_path))
    print(f"✓ Saved: {xgb_path}")
    
    # Save comparison
    comparison_df = pd.DataFrame([lr_metrics, xgb_metrics])
    comparison_path = MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved: {comparison_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'models': {
            'logistic_regression': lr_metrics,
            'xgboost': xgb_metrics
        },
        'best_model': 'xgboost' if xgb_metrics['test_auc'] > lr_metrics['test_auc'] else 'logistic_regression',
        'performance': {
            'auc_roc': max(lr_metrics['test_auc'], xgb_metrics['test_auc']),
            'gini': max(lr_metrics['test_gini'], xgb_metrics['test_gini']),
            'ks_statistic': max(lr_metrics['test_ks'], xgb_metrics['test_ks'])
        }
    }
    
    save_json(metadata, MODELS_DIR / "model_metadata.json")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute complete training pipeline."""
    
    print_section("PHASE 4: MODEL TRAINING & EVALUATION", width=80)
    
    # Step 1: Load data
    X, y = load_processed_data()
    
    # Step 2: Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_split(X, y)
    
    # Step 3: Train Logistic Regression
    lr_model, lr_train_info = train_logistic_regression(
        X_train, y_train, X_val, y_val, use_grid_search=False
    )
    
    # Step 4: Train XGBoost
    xgb_model, xgb_train_info = train_xgboost(
        X_train, y_train, X_val, y_val, use_grid_search=False
    )
    
    # Step 5: Comprehensive evaluation
    lr_metrics = evaluate_model_comprehensive(
        lr_model, "Logistic Regression", X_val, y_val, X_test, y_test
    )
    
    xgb_metrics = evaluate_model_comprehensive(
        xgb_model, "XGBoost", X_val, y_val, X_test, y_test
    )
    
    # Step 6: Model comparison
    print_section("MODEL COMPARISON")
    comparison = pd.DataFrame([lr_metrics, xgb_metrics])
    print(comparison[['model_name', 'test_auc', 'test_gini', 'test_ks', 'test_f1']].to_string(index=False))
    
    # Determine best model
    if xgb_metrics['test_auc'] > lr_metrics['test_auc']:
        print(f"\n🏆 Best Model: XGBoost (AUC: {xgb_metrics['test_auc']:.4f})")
    else:
        print(f"\n🏆 Best Model: Logistic Regression (AUC: {lr_metrics['test_auc']:.4f})")
    
    # Step 7: Save models
    save_models(lr_model, xgb_model, lr_metrics, xgb_metrics)
    
    # Final summary
    print_section("✓ PHASE 4 COMPLETE", width=80)
    print("Summary:")
    print(f"  ✓ Models trained and validated")
    print(f"  ✓ Performance metrics calculated")
    print(f"  ✓ Models saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Review model comparison: cat models/model_comparison.csv")
    print("  2. Run explainability: python src/5_shap_analysis.py")
    print("  3. Start API: python app/main.py")


if __name__ == "__main__":
    main()


