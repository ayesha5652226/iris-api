from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# your existing imports...
from pydantic import BaseModel, conlist
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Iris Classifier API", version="1.0")

# 👇 Initialize templates directory
templates = Jinja2Templates(directory="templates")  # assumes your HTML is in /templates/

# 👇 input model for JSON API
class PointInput(BaseModel):
    point: conlist(float, 4)  # Requires exactly 4 float values

# 👇 load or train model
MODEL_PATH = "model.joblib"

def train_default_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    pipeline.fit(X, y)
    return {"pipeline": pipeline, "target_names": iris.target_names.tolist()}

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        data = train_default_model()
        joblib.dump(data, MODEL_PATH)
        return data

model_bundle = load_model()
pipeline = model_bundle["pipeline"]
target_names = model_bundle["target_names"]

# 👇 Updated root route to serve HTML
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 👇 JSON POST API
@app.post("/predict")
def predict(input: PointInput):
    X = [input.point]
    try:
        pred_idx = int(pipeline.predict(X)[0])
        proba = pipeline.predict_proba(X)[0].tolist() if hasattr(pipeline, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    probabilities = {target_names[i]: float(p) for i, p in enumerate(proba)} if proba else None

    return {
        "prediction": target_names[pred_idx],
        "probabilities": probabilities
    }
