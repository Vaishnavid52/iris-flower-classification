"""
Iris Flower Classification — Phase 3: FastAPI Backend
======================================================
Endpoints:
  GET  /               → health check
  POST /predict        → predict species + confidence + all probabilities
  GET  /dataset        → return full dataset for frontend 3D visualization
  GET  /model-stats    → model performance metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌸 Iris Flower Classifier API",
    description="Phase 3 — FastAPI backend for Iris classification with SVM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow React frontend on any port
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load or train model ───────────────────────────────────────────────────────
SPECIES = ["Setosa", "Versicolor", "Virginica"]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

MODEL_PATH  = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

iris   = load_iris()
X, y   = iris.data, iris.target

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded saved model OK")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel="linear", C=10, probability=True, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Trained and saved new model OK")

X_scaled = scaler.transform(X)

# ── Schemas ───────────────────────────────────────────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=4.0, le=8.0, example=5.1,
                                 description="Sepal length in cm (4.0–8.0)")
    sepal_width:  float = Field(..., ge=1.5, le=5.0, example=3.5,
                                 description="Sepal width in cm (1.5–5.0)")
    petal_length: float = Field(..., ge=1.0, le=7.0, example=1.4,
                                 description="Petal length in cm (1.0–7.0)")
    petal_width:  float = Field(..., ge=0.1, le=2.6, example=0.2,
                                 description="Petal width in cm (0.1–2.6)")

class PredictionResponse(BaseModel):
    species:      str
    species_id:   int
    confidence:   float
    probabilities: dict
    input_scaled: list
    nearest_neighbors: list

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "🌸 Iris API is live",
        "version": "1.0.0",
        "endpoints": ["/predict", "/dataset", "/model-stats", "/docs"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: IrisInput):
    """
    Predict the Iris species from 4 measurements.
    Returns species, confidence %, all class probabilities,
    scaled input values, and 3 nearest neighbors from the dataset.
    """
    raw = np.array([[data.sepal_length, data.sepal_width,
                     data.petal_length, data.petal_width]])
    scaled = scaler.transform(raw)

    pred_id   = int(model.predict(scaled)[0])
    proba     = model.predict_proba(scaled)[0]
    confidence = float(proba[pred_id])

    # Find 3 nearest neighbors for frontend visualization
    dists = np.linalg.norm(X_scaled - scaled, axis=1)
    nn_idx = np.argsort(dists)[:3].tolist()
    neighbors = [
        {
            "index": int(i),
            "species": SPECIES[int(y[i])],
            "distance": round(float(dists[i]), 4),
            "features": {f: round(float(X[i][j]), 2) for j, f in enumerate(FEATURES)}
        }
        for i in nn_idx
    ]

    return PredictionResponse(
        species=SPECIES[pred_id],
        species_id=pred_id,
        confidence=round(confidence * 100, 2),
        probabilities={
            SPECIES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        },
        input_scaled=scaled[0].tolist(),
        nearest_neighbors=neighbors,
    )


@app.get("/dataset", tags=["Data"])
def get_dataset():
    """
    Return the full Iris dataset for 3D/4D frontend visualization.
    Includes PCA-reduced 3D coordinates for each point.
    """
    from sklearn.decomposition import PCA

    pca3 = PCA(n_components=3)
    X_3d = pca3.fit_transform(X_scaled)

    points = []
    for i in range(len(X)):
        points.append({
            "index":        i,
            "species":      SPECIES[int(y[i])],
            "species_id":   int(y[i]),
            "sepal_length": round(float(X[i][0]), 2),
            "sepal_width":  round(float(X[i][1]), 2),
            "petal_length": round(float(X[i][2]), 2),
            "petal_width":  round(float(X[i][3]), 2),
            "pca_x":        round(float(X_3d[i][0]), 4),
            "pca_y":        round(float(X_3d[i][1]), 4),
            "pca_z":        round(float(X_3d[i][2]), 4),
        })

    return {
        "total": len(points),
        "species_counts": {s: int(np.sum(y == i)) for i, s in enumerate(SPECIES)},
        "pca_variance_explained": [round(float(v), 4) for v in pca3.explained_variance_ratio_],
        "points": points,
    }


@app.get("/model-stats", tags=["Model"])
def model_stats():
    """
    Return model performance metrics, feature importances (via permutation),
    and cross-validation scores.
    """
    from sklearn.inspection import permutation_importance

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    return {
        "model_type": type(model).__name__,
        "parameters": {str(k): str(v) for k, v in model.get_params().items()},
        "metrics": {
            "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "cv_mean":        round(float(cv_scores.mean()), 4),
            "cv_std":         round(float(cv_scores.std()), 4),
            "precision":      round(precision_score(y_test, y_pred, average="weighted"), 4),
            "recall":         round(recall_score(y_test, y_pred, average="weighted"), 4),
            "f1_score":       round(f1_score(y_test, y_pred, average="weighted"), 4),
        },
        "feature_importance": {
            FEATURES[i]: round(float(perm.importances_mean[i]), 4)
            for i in range(4)
        },
        "cv_scores_per_fold": [round(float(s), 4) for s in cv_scores],
    }
