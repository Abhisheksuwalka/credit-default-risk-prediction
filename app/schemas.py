from pydantic import BaseModel, Field
from typing import Optional

class LoanApplication(BaseModel):
    """Loan application input schema"""
    revenue: float = Field(
        ..., 
        description="Annual income in USD",
        gt=0
    )
    loan_amnt: float = Field(
        ..., 
        description="Loan amount requested",
        gt=0
    )
    fico_n: int = Field(
        ..., 
        description="FICO credit score",
        ge=300,
        le=850
    )
    dti_n: float = Field(
        ..., 
        description="Debt-to-income ratio",
        ge=0,
        le=1
    )
    emp_length_numeric: int = Field(
        ..., 
        description="Employment length in years",
        ge=0,
        le=10
    )

    class Config:
        schema_extra = {
            "example": {
                "revenue": 50000,
                "loan_amnt": 10000,
                "fico_n": 700,
                "dti_n": 0.25,
                "emp_length_numeric": 5
            }
        }

class PredictionResponse(BaseModel):
    """Prediction output schema"""
    probability_of_default: float
    probability_percentage: str
    risk_category: str
    confidence_score: float
    recommendation: str
    top_risk_factors: list = []

class ModelInfo(BaseModel):
    """Model metadata"""
    model_name: str
    version: str
    algorithm: str
    features_count: int
    training_samples: int
    default_rate: float
    performance_auc: float