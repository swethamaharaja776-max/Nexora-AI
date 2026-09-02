"""
Database CRUD operations for Nexora AI.
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session, joinedload

from . import models, schemas


def create_decision(db: Session, payload: schemas.DecisionCreate) -> models.Decision:
    decision = models.Decision(
        title=payload.title,
        problem_statement=payload.problem_statement,
        category=payload.category,
        status="draft",
    )
    db.add(decision)
    db.flush()

    for f in payload.factors:
        db.add(models.Factor(decision_id=decision.id, name=f.name, weight=f.weight, direction=f.direction))

    for o in payload.options:
        db.add(models.Option(decision_id=decision.id, name=o.name, values=o.values, series=o.series))

    db.commit()
    db.refresh(decision)
    return decision


def get_decision(db: Session, decision_id: str) -> Optional[models.Decision]:
    return (
        db.query(models.Decision)
        .options(joinedload(models.Decision.factors), joinedload(models.Decision.options), joinedload(models.Decision.analysis))
        .filter(models.Decision.id == decision_id)
        .first()
    )


def list_decisions(db: Session):
    return (
        db.query(models.Decision)
        .options(joinedload(models.Decision.analysis))
        .order_by(models.Decision.created_at.desc())
        .all()
    )


def delete_decision(db: Session, decision_id: str) -> bool:
    decision = db.query(models.Decision).filter(models.Decision.id == decision_id).first()
    if not decision:
        return False
    db.delete(decision)
    db.commit()
    return True


def save_analysis(db: Session, decision_id: str, results: dict, insights: dict, ai_source: str) -> models.AnalysisResult:
    existing = db.query(models.AnalysisResult).filter(models.AnalysisResult.decision_id == decision_id).first()
    if existing:
        db.delete(existing)
        db.flush()

    analysis = models.AnalysisResult(
        decision_id=decision_id,
        metrics=results["metrics"],
        trends=results["trends"],
        correlations=results["correlations"],
        risk=results["risk"],
        recommendations=results["recommendations"],
        insights=insights,
        comparison=results["comparison"],
        ai_source=ai_source,
    )
    db.add(analysis)

    decision = db.query(models.Decision).filter(models.Decision.id == decision_id).first()
    if decision:
        decision.status = "analyzed"

    db.commit()
    db.refresh(analysis)
    return analysis
