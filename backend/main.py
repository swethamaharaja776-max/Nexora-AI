from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import io
import json
import os
import uuid

# ---------------- APPLICATION ----------------

app = FastAPI(
    title="NEXORA AI API",
    description="Intelligent Decision Support System",
    version="1.0.0"
)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------- DATABASE ----------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nexora.db")

engine_options = {}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(36), index=True)
    filename = Column(String(255))
    rows_count = Column(Integer)
    columns_count = Column(Integer)
    summary = Column(Text)


Base.metadata.create_all(bind=engine)

# ---------------- TEMPORARY DATASET STORAGE ----------------

datasets = {}


# ---------------- REQUEST MODELS ----------------

class PredictionRequest(BaseModel):
    dataset_id: str
    input_column: str
    target_column: str
    input_value: float


# ---------------- HEALTH CHECK ----------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "application": "NEXORA AI",
        "version": "1.0.0"
    }


# ---------------- UPLOAD DATASET ----------------

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    content = await file.read(5 * 1024 * 1024 + 1)

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 5 MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file."
        )

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="The dataset is empty."
        )

    if len(df.columns) > 100 or len(df) > 20000:
        raise HTTPException(
            status_code=400,
            detail="Dataset exceeds the supported row or column limit."
        )

    df.columns = [str(col) for col in df.columns]

    if not df.columns.is_unique:
        raise HTTPException(
            status_code=400,
            detail="Column names must be unique."
        )

    dataset_id = str(uuid.uuid4())

    datasets[dataset_id] = df

    return {
        "dataset_id": dataset_id,
        "filename": os.path.basename(file.filename),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isna().sum().sum()),
        "message": "Dataset uploaded successfully."
    }


# ---------------- LIST DATASETS ----------------

@app.get("/api/datasets")
def list_datasets():
    return [
        {
            "dataset_id": dataset_id,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist()
        }
        for dataset_id, df in datasets.items()
    ]


# ---------------- DATASET DETAILS ----------------

@app.get("/api/datasets/{dataset_id}")
def dataset_details(dataset_id: str):

    df = datasets.get(dataset_id)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload it again."
        )

    preview = json.loads(
        df.head(100).to_json(orient="records", date_format="iso")
    )

    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "preview": preview,
        "missing_values": int(df.isna().sum().sum())
    }


# ---------------- DATA ANALYSIS ----------------

@app.post("/api/datasets/{dataset_id}/analyze")
def analyze_dataset(dataset_id: str):

    df = datasets.get(dataset_id)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Upload the CSV again."
        )

    numeric = df.select_dtypes(include=np.number)

    statistics = {}

    for col in numeric.columns:
        series = numeric[col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        statistics[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()) if len(series) else None,
            "median": float(series.median()) if len(series) else None,
            "minimum": float(series.min()) if len(series) else None,
            "maximum": float(series.max()) if len(series) else None,
            "std": float(series.std()) if len(series) > 1 else 0.0
        }

    result = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": df.select_dtypes(
            exclude=np.number
        ).columns.tolist(),
        "statistics": statistics,
        "correlations": json.loads(
            numeric.corr().round(4).to_json()
        ) if len(numeric.columns) else {}
    }

    db = SessionLocal()

    try:
        db.add(
            AnalysisHistory(
                dataset_id=dataset_id,
                filename="Uploaded CSV",
                rows_count=int(df.shape[0]),
                columns_count=int(df.shape[1]),
                summary=json.dumps(result, allow_nan=False)
            )
        )
        db.commit()
    finally:
        db.close()

    return result


# ---------------- MACHINE LEARNING ----------------

@app.post("/api/datasets/{dataset_id}/predict")
def predict(dataset_id: str, request: PredictionRequest):

    if dataset_id != request.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Dataset ID mismatch."
        )

    df = datasets.get(dataset_id)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    if request.input_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Input column does not exist."
        )

    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Target column does not exist."
        )

    if request.input_column == request.target_column:
        raise HTTPException(
            status_code=400,
            detail="Choose two different columns."
        )

    try:
        x = pd.to_numeric(
            df[request.input_column],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        y = pd.to_numeric(
            df[request.target_column],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid = x.notna() & y.notna()

        x = x[valid].to_numpy().reshape(-1, 1)
        y = y[valid].to_numpy()

        if len(x) < 5:
            raise HTTPException(
                status_code=400,
                detail="At least five valid rows are required."
            )

        if len(np.unique(x)) < 2:
            raise HTTPException(
                status_code=400,
                detail="Input column must contain varying values."
            )

        if len(np.unique(y)) < 2:
            raise HTTPException(
                status_code=400,
                detail="Target column must contain varying values."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        prediction = float(
            model.predict(
                np.array([[request.input_value]], dtype=float)
            )[0]
        )

        return {
            "input_column": request.input_column,
            "target_column": request.target_column,
            "input_value": request.input_value,
            "predicted_value": prediction,
            "test_rows": len(y_test),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "model": "Linear Regression",
            "note": "Predictions are estimates, not guarantees."
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}"
        )


# ---------------- HISTORY ----------------

@app.get("/api/history")
def get_history():

    db = SessionLocal()

    try:
        history = (
            db.query(AnalysisHistory)
            .order_by(AnalysisHistory.id.desc())
            .limit(100)
            .all()
        )

        return [
            {
                "id": item.id,
                "dataset_id": item.dataset_id,
                "filename": item.filename,
                "rows": item.rows_count,
                "columns": item.columns_count,
                "summary": json.loads(item.summary)
            }
            for item in history
        ]

    finally:
        db.close()


# ---------------- DOWNLOAD REPORT ----------------

@app.get("/api/reports/{dataset_id}")
def download_report(dataset_id: str):

    df = datasets.get(dataset_id)

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    report = io.StringIO()
    report.write("NEXORA AI - DATA ANALYSIS REPORT\n")
    report.write("=" * 35 + "\n\n")
    report.write(f"Rows: {len(df)}\n")
    report.write(f"Columns: {len(df.columns)}\n")
    report.write(
        f"Missing values: {int(df.isna().sum().sum())}\n\n"
    )

    report.write("STATISTICAL SUMMARY\n")
    report.write(df.describe(include="all").to_string())
    report.write("\n\nMISSING VALUES\n")
    report.write(df.isna().sum().to_string())

    output = io.BytesIO(report.getvalue().encode("utf-8"))

    return StreamingResponse(
        output,
        media_type="text/plain",
        headers={
            "Content-Disposition": 'attachment; filename="nexora-report.txt"'
        }
      )
