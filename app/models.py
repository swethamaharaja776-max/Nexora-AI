"""
SQLAlchemy ORM models for Nexora AI.

A Decision is the top-level object a user creates. It owns:
  - Factors      (weighted criteria the user cares about, e.g. "Cost", "Risk")
  - Options      (the choices being decided between, e.g. "Vendor A")
  - DataPoints   (numeric supporting data per option/factor, used for analysis)
  - AnalysisResult (the persisted output of running the analysis engine)
"""
import datetime
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, ForeignKey, DateTime, Text, JSON
)
from sqlalchemy.orm import relationship

from .database import Base


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(String, primary_key=True, default=gen_id)
    title = Column(String, nullable=False)
    problem_statement = Column(Text, nullable=False)
    category = Column(String, default="General")
    status = Column(String, default="draft")  # draft | analyzed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    factors = relationship("Factor", back_populates="decision", cascade="all, delete-orphan")
    options = relationship("Option", back_populates="decision", cascade="all, delete-orphan")
    analysis = relationship("AnalysisResult", back_populates="decision", uselist=False, cascade="all, delete-orphan")


class Factor(Base):
    __tablename__ = "factors"

    id = Column(String, primary_key=True, default=gen_id)
    decision_id = Column(String, ForeignKey("decisions.id"))
    name = Column(String, nullable=False)
    weight = Column(Float, default=1.0)  # 0-1, relative importance
    direction = Column(String, default="higher_better")  # higher_better | lower_better

    decision = relationship("Decision", back_populates="factors")


class Option(Base):
    __tablename__ = "options"

    id = Column(String, primary_key=True, default=gen_id)
    decision_id = Column(String, ForeignKey("decisions.id"))
    name = Column(String, nullable=False)
    # values: {factor_name: numeric_value}
    values = Column(JSON, default=dict)
    # optional time series data for trend analysis: {factor_name: [numbers]}
    series = Column(JSON, default=dict)

    decision = relationship("Decision", back_populates="options")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True, default=gen_id)
    decision_id = Column(String, ForeignKey("decisions.id"), unique=True)

    metrics = Column(JSON, default=dict)          # key metrics per option/factor
    trends = Column(JSON, default=dict)            # trend direction/slope per factor
    correlations = Column(JSON, default=dict)      # factor-to-factor correlation matrix
    risk = Column(JSON, default=dict)               # risk score + level + contributing factors
    recommendations = Column(JSON, default=list)    # list of recommendation objects
    insights = Column(JSON, default=dict)            # AI insights panel content
    comparison = Column(JSON, default=dict)          # option scoring for comparison view
    ai_source = Column(String, default="demo")        # "llm" or "demo"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    decision = relationship("Decision", back_populates="analysis")
