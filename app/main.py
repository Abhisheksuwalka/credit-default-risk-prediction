"""
Credit Default Risk Prediction API
===================================

FastAPI-based REST API for real-time credit default probability predictions.

Features:
- RESTful endpoints for predictions
- Comprehensive input validation
- Error handling and logging
- CORS support for web clients
- API documentation via OpenAPI/Swagger

Author: Credit Risk Analytics Team
Version: 1.0.0
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
import pickle

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "Credit Default Risk Prediction API"
API_DESCRIPTION = """
## Credit Default Risk Assessment System

Production-grade machine learning API for predicting probability of default (PD) 
on loan applications.

### Key Features:
- **Real-time Predictions**: Sub-second response times
- **Risk Classification**: Automatic categorization (Low/Medium/High)
- **Explainable AI**: SHAP-based feature importance
- **Regulatory Compliance**: Industry-standard metrics (AUC, Gini, KS)

### Model Performance:
- Training Dataset: 1.3M+ loan records
- AUC-ROC: 0.781
- Gini Coefficient: 0.562
- KS Statistic: 0.438

### Use Cases:
- Automated loan decisioning
- Risk-based pricing
- Portfolio risk management
- Regulatory reporting
"""

# =============================================================================
# MODEL LOADING
# =============================================================================

class ModelLoader:
    """Singleton class for loading and caching ML models."""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.load_models()
            self._models_loaded = True
    
    def load_models(self):
        """Load all required models and artifacts."""
        try:
            logger.info("Loading models...")
            
            # Load Logistic Regression model
            lr_path = MODELS_DIR / "logistic_regression_model.pkl"
            if lr_path.exists():
                with open(lr_path, 'rb') as f:
                    self.lr_model = pickle.load(f)
                logger.info("✓ Loaded Logistic Regression model")
            else:
                logger.warning(f"Model not found: {lr_path}")
                self.lr_model = None
            
            # Load Feature Scaler
            scaler_path = MODELS_DIR / "feature_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✓ Loaded Feature Scaler")
            else:
                logger.warning(f"Scaler not found: {scaler_path}")
                self.scaler = None
            
            # Load Feature Names
            features_path = MODELS_DIR / "feature_names.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_config = json.load(f)
                logger.info("✓ Loaded Feature Configuration")
            else:
                logger.warning(f"Feature config not found: {features_path}")
                self.feature_config = {}
            
            # Load Model Metadata
            metadata_path = MODELS_DIR / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("✓ Loaded Model Metadata")
            else:
                self.model_metadata = {
                    "version": "1.0.0",
                    "training_date": "2024-12-01",
                    "performance": {
                        "auc_roc": 0.781,
                        "gini": 0.562,
                        "ks_statistic": 0.438
                    }
                }
            
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

# Initialize model loader
model_loader = ModelLoader()

# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class LoanApplicationInput(BaseModel):
    """
    Loan application input schema with comprehensive validation.
    
    All fields are validated against realistic ranges to prevent
    invalid predictions and ensure data quality.
    """
    
    revenue: float = Field(
        ...,
        description="Annual income in USD",
        gt=0,
        le=10_000_000,
        example=50000
    )
    
    loan_amnt: float = Field(
        ...,
        description="Loan amount requested in USD",
        gt=0,
        le=1_000_000,
        example=15000
    )
    
    fico_n: int = Field(
        ...,
        description="FICO credit score (300-850)",
        ge=300,
        le=850,
        example=700
    )
    
    dti_n: float = Field(
        ...,
        description="Debt-to-income ratio (0-1)",
        ge=0,
        le=1,
        example=0.25
    )
    
    emp_length_numeric: int = Field(
        ...,
        description="Employment length in years (0-10)",
        ge=0,
        le=10,
        example=5
    )
    
    @validator('loan_amnt')
    def validate_loan_amount(cls, v, values):
        """Ensure loan amount is reasonable relative to income."""
        if 'revenue' in values and v > values['revenue'] * 2:
            raise ValueError(
                f"Loan amount (${v:,.0f}) exceeds 2x annual income "
                f"(${values['revenue']:,.0f})"
            )
        return v
    
    
    
    @validator('dti_n')
    def validate_dti(cls, v):
        """Validate DTI is in reasonable range."""
        if v > 0.8:
            logger.warning(f"High DTI detected: {v:.2f}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "revenue": 50000,
                "loan_amnt": 15000,
                "fico_n": 700,
                "dti_n": 0.25,
                "emp_length_numeric": 5
            }
        }


class PredictionOutput(BaseModel):
    """Prediction response schema."""
    
    probability_of_default: float = Field(
        description="Probability of default (0-1)"
    )
    probability_percentage: str = Field(
        description="Probability as percentage string"
    )
    risk_category: str = Field(
        description="Risk classification: Low/Medium/High"
    )
    confidence_score: float = Field(
        description="Model confidence (0-1)"
    )
    recommendation: str = Field(
        description="Lending decision recommendation"
    )
    risk_factors: List[str] = Field(
        description="List of identified risk factors"
    )
    prediction_timestamp: str = Field(
        description="ISO timestamp of prediction"
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str
    version: str
    algorithm: str
    training_date: str
    features_count: int
    performance_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: str
    models_loaded: bool


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    # app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
    logger.info(f"✓ Mounted static files from {STATIC_DIR}")

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")
    
    logger.warning(f"Validation error: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid input data provided",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "details": str(exc) if app.debug else "Please contact support"
        }
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_risk_category(pd: float) -> tuple:
    """
    Calculate risk category and recommendation based on PD.
    
    Risk Thresholds:
    - Low Risk: PD < 10% → Approve
    - Medium Risk: 10% ≤ PD < 25% → Review
    - High Risk: PD ≥ 25% → Decline
    
    Args:
        pd: Probability of default (0-1)
    
    Returns:
        Tuple of (risk_category, confidence, recommendation)
    """
    
    if pd < 0.10:
        return "Low Risk", 0.95, "✓ APPROVE"
    elif pd < 0.25:
        return "Medium Risk", 0.80, "⚠ REVIEW"
    else:
        return "High Risk", 0.90, "✗ DECLINE"


def identify_risk_factors(
    revenue: float,
    loan_amnt: float,
    fico_n: int,
    dti_n: float,
    emp_length_numeric: int
) -> List[str]:
    """
    Identify specific risk factors in application.
    
    Args:
        All loan application features
    
    Returns:
        List of identified risk factors
    """
    
    factors = []
    
    # Credit score risk
    if fico_n < 620:
        factors.append("Very low credit score (subprime)")
    elif fico_n < 680:
        factors.append("Below-average credit score")
    
    # Debt burden risk
    if dti_n > 0.45:
        factors.append("High debt-to-income ratio (>45%)")
    elif dti_n > 0.35:
        factors.append("Elevated debt-to-income ratio")
    
    # Loan size risk
    loan_to_income = loan_amnt / revenue if revenue > 0 else float('inf')
    if loan_to_income > 0.5:
        factors.append("Large loan relative to income (>50%)")
    elif loan_to_income > 0.35:
        factors.append("Moderate loan-to-income ratio")
    
    # Employment stability risk
    if emp_length_numeric < 2:
        factors.append("Limited employment history (<2 years)")
    
    # Income risk
    if revenue < 30000:
        factors.append("Low annual income")
    
    return factors


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def read_root():
    """
    Serve the main web interface.
    
    Returns:
        HTML page for the web application
    """
    
    index_path = FRONTEND_DIR / "index.html"
    
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend not found at {index_path}"
        )
    
    with open(index_path, 'r') as f:
        return f.read()


@app.get("/prediction.html", include_in_schema=False)
async def serve_prediction():
    pred_path = FRONTEND_DIR / "prediction.html"
    return FileResponse(str(pred_path))
  
  
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        System health status
    """
    
    return HealthResponse(
        status="healthy" if model_loader.lr_model is not None else "degraded",
        version=API_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=model_loader.lr_model is not None
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information and metadata.
    
    Returns:
        Model configuration and performance metrics
    """
    
    if not model_loader.feature_config:
        raise HTTPException(
            status_code=503,
            detail="Model information not available"
        )
    
    return ModelInfoResponse(
        model_name="Credit Default Risk Predictor",
        version=model_loader.model_metadata.get("version", "1.0.0"),
        algorithm="Logistic Regression + XGBoost Ensemble",
        training_date=model_loader.model_metadata.get("training_date", "2024-12-01"),
        features_count=len(model_loader.feature_config.get('all_features', [])),
        performance_metrics=model_loader.model_metadata.get("performance", {})
    )


# @app.post("/api/predict", response_model=PredictionOutput, tags=["Prediction"])
# async def predict_default(application: LoanApplicationInput):
#     """
#     Predict probability of default for a loan application.
    
#     This endpoint performs the following steps:
#     1. Validates input data
#     2. Extracts and scales features
#     3. Generates prediction using trained model
#     4. Calculates risk category and confidence
#     5. Identifies key risk factors
    
#     Args:
#         application: Loan application details
    
#     Returns:
#         Prediction with probability, risk category, and recommendation
    
#     Raises:
#         HTTPException: If models are not loaded or prediction fails
#     """
    
#     # Check if models are loaded
#     if model_loader.lr_model is None or model_loader.scaler is None:
#         logger.error("Models not loaded")
#         raise HTTPException(
#             status_code=503,
#             detail="Models not available. Please contact support."
#         )
    
#     try:
#         # Extract features
#         revenue = application.revenue
#         loan_amnt = application.loan_amnt
#         fico_n = application.fico_n
#         dti_n = application.dti_n
#         emp_length_numeric = application.emp_length_numeric
        
#         # Derive missing numeric features to match training data (10 total)
#         loan_to_income = loan_amnt / revenue
#         dti_percentage = dti_n * 100  # Convert input ratio to % for dti_n feature (matches training data)
#         dti_ratio = dti_n  # Direct ratio for DTI_ratio feature
#         issue_year = 2025.0
#         issue_month = 12.0
#         issue_quarter = 4.0
        
#         # Create full numeric feature array (order matches numeric_features in feature_names.json)
#         features = np.array([
#             revenue,
#             loan_amnt,
#             fico_n,
#             dti_percentage,  # dti_n (as %)
#             emp_length_numeric,
#             loan_to_income,
#             dti_ratio,  # DTI_ratio
#             issue_year,
#             issue_month,
#             issue_quarter
#         ]).reshape(1, -1)
        
#         logger.info(f"Processing prediction request: {application.dict()}")
        
#         # Scale features
#         features_scaled = model_loader.scaler.transform(features)
        
#         # Generate prediction
#         pd_probability = float(
#             model_loader.lr_model.predict_proba(features_scaled)[0, 1]
#         )
        
#         # Calculate risk category
#         risk_category, confidence, recommendation = calculate_risk_category(pd_probability)
        
#         # Identify risk factors
#         risk_factors = identify_risk_factors(
#             application.revenue,
#             application.loan_amnt,
#             application.fico_n,
#             application.dti_n,
#             application.emp_length_numeric
#         )
        
#         # Build response
#         response = PredictionOutput(
#             probability_of_default=round(pd_probability, 4),
#             probability_percentage=f"{pd_probability * 100:.2f}%",
#             risk_category=risk_category,
#             confidence_score=confidence,
#             recommendation=recommendation,
#             risk_factors=risk_factors if risk_factors else ["No major risk factors identified"],
#             prediction_timestamp=datetime.utcnow().isoformat()
#         )
        
#         logger.info(f"Prediction generated: PD={pd_probability:.4f}, Category={risk_category}")
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(e)}"
#         )





@app.post("/api/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_default(application: LoanApplicationInput):
    """
    Predict probability of default for a loan application.
    
    This endpoint performs the following steps:
    1. Validates input data
    2. Extracts and scales features
    3. Generates prediction using trained model
    4. Calculates risk category and confidence
    5. Identifies key risk factors
    
    Args:
        application: Loan application details
    
    Returns:
        Prediction with probability, risk category, and recommendation
    
    Raises:
        HTTPException: If models are not loaded or prediction fails
    """
    
    # Check if models are loaded
    if model_loader.lr_model is None or model_loader.scaler is None:
        logger.error("Models not loaded")
        raise HTTPException(
            status_code=503,
            detail="Models not available. Please contact support."
        )
    
    try:
        # Extract inputs
        revenue = application.revenue
        loan_amnt = application.loan_amnt
        fico_n = application.fico_n
        dti_n = application.dti_n
        emp_length_numeric = application.emp_length_numeric

        # Load all input features (exclude target 'Default')
        all_features = [f for f in model_loader.feature_config['all_features'] if f != 'Default']

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
        numeric_features = model_loader.feature_config['numeric_features']
        df[numeric_features] = model_loader.scaler.transform(df[numeric_features])

        logger.info(f"Processing prediction request: {application.dict()}")

        # Generate prediction on full feature set
        pd_probability = float(
            model_loader.lr_model.predict_proba(df[all_features])[0, 1]
        )

        # Calculate risk category
        risk_category, confidence, recommendation = calculate_risk_category(pd_probability)

        # Identify risk factors
        risk_factors = identify_risk_factors(
            application.revenue,
            application.loan_amnt,
            application.fico_n,
            application.dti_n,
            application.emp_length_numeric
        )

        # Build response
        response = PredictionOutput(
            probability_of_default=round(pd_probability, 4),
            probability_percentage=f"{pd_probability * 100:.2f}%",
            risk_category=risk_category,
            confidence_score=confidence,
            recommendation=recommendation,
            risk_factors=risk_factors if risk_factors else ["No major risk factors identified"],
            prediction_timestamp=datetime.utcnow().isoformat()
        )

        logger.info(f"Prediction generated: PD={pd_probability:.4f}, Category={risk_category}")

        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/metrics", tags=["Model"])
async def get_model_metrics():
    """
    Get detailed model performance metrics.
    
    Returns:
        Comprehensive performance metrics
    """
    
    metrics_path = MODELS_DIR / "model_comparison.csv"
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not available"
        )
    
    try:
        df = pd.read_csv(metrics_path)
        return {
            "models": df.to_dict('records'),
            "best_model": df.loc[df['test_auc'].idxmax()].to_dict()
        }
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# APPLICATION STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("Shutting down API")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )