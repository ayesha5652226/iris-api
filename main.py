from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.joblib"

app = FastAPI(title="Iris Classifier API", version="1.0")

templates = Jinja2Templates(directory="templates")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def train_default_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    pipeline.fit(X, y)
    return {"pipeline": pipeline, "target_names": iris.target_names.tolist()}

def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data
    else:
        data = train_default_model()
        joblib.dump(data, MODEL_PATH)
        return data

model_bundle = load_model()
pipeline = model_bundle["pipeline"]
target_names = model_bundle["target_names"]

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(features: IrisFeatures):
    X = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    try:
        pred_idx = int(pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    proba = pipeline.predict_proba(X)[0].tolist() if hasattr(pipeline, "predict_proba") else None

    if proba is not None:
        probs = {name: float(p) for name, p in zip(target_names, proba)}
    else:
        probs = None

    return {"prediction": target_names[pred_idx], "probabilities": probs}
