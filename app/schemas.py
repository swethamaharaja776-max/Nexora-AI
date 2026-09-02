"""
Pydantic request/response schemas for the Nexora AI API.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class FactorIn(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=0, le=1)
    direction: str = "higher_better"  # higher_better | lower_better

    @field_validator("direction")
    @classmethod
    def valid_direction(cls, v):
        if v not in ("higher_better", "lower_better"):
            raise ValueError("direction must be 'higher_better' or 'lower_better'")
        return v


class OptionIn(BaseModel):
    name: str
    values: Dict[str, float] = Field(default_factory=dict)
    series: Dict[str, List[float]] = Field(default_factory=dict)


class DecisionCreate(BaseModel):
    title: str
    problem_statement: str
    category: str = "General"
    factors: List[FactorIn] = Field(default_factory=list)
    options: List[OptionIn] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()

    @field_validator("problem_statement")
    @classmethod
    def problem_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("problem_statement cannot be empty")
        return v.strip()


class FactorOut(BaseModel):
    id: str
    name: str
    weight: float
    direction: str
    model_config = {"from_attributes": True}


class OptionOut(BaseModel):
    id: str
    name: str
    values: Dict[str, Any]
    series: Dict[str, Any]
    model_config = {"from_attributes": True}


class DecisionOut(BaseModel):
    id: str
    title: str
    problem_statement: str
    category: str
    status: str
    created_at: datetime
    updated_at: datetime
    factors: List[FactorOut] = []
    options: List[OptionOut] = []
    model_config = {"from_attributes": True}


class DecisionSummary(BaseModel):
    id: str
    title: str
    category: str
    status: str
    created_at: datetime
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    overall_score: Optional[float] = None
    model_config = {"from_attributes": True}


class AnalysisOut(BaseModel):
    id: str
    decision_id: str
    metrics: Dict[str, Any]
    trends: Dict[str, Any]
    correlations: Dict[str, Any]
    risk: Dict[str, Any]
    recommendations: List[Any]
    insights: Dict[str, Any]
    comparison: Dict[str, Any]
    ai_source: str
    created_at: datetime
    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    total_analyses: int
    high_risk_count: int
    average_score: float
    recent_analyses: List[DecisionSummary]
    recent_recommendations: List[Dict[str, Any]]
