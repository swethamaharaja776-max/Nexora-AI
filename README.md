# Nexora AI

**Intelligent Decision Support System**
*Turn Data Into Better Decisions.*

Nexora AI analyzes structured data you provide, detects patterns, scores risk transparently, and produces explainable, evidence-based recommendations.

---

## Project Structure
Nexora-AI/
├── app/
│   ├── init.py
│   ├── main.py              # FastAPI app entrypoint
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # Decision, Factor, Option, AnalysisResult
│   ├── schemas.py           # Pydantic request/response models
│   ├── crud.py               # DB operations
│   ├── ai_service.py         # LLM abstraction + demo fallback
│   ├── analysis_engine.py    # deterministic pandas/NumPy analysis
│   └── routers/
│       ├── init.py
│       ├── decisions.py      # decision CRUD + analyze/insights/recommendations/risk
│       ├── dashboard.py      # aggregate dashboard stats
│       └── reports.py        # PDF report generation
├── frontend/
│   ├── index.html            # functional single-page app (vanilla JS, real API calls)
│   └── index.css
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
---

## How This Runs

The backend is a standard FastAPI package rooted at `app/`, run **from the project root**:

```bash
uvicorn app.main:app --reload --port 8000
