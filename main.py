from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.joblib"

app = FastAPI(title="Iris Classifier API", version="1.0")

# ✅ Allow CORS (for frontend on another domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Accept 'point': list of 4 floats
class PointInput(BaseModel):
    point: conlist(float, min_items=4, max_items=4)

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

@app.get("/")
def root():
    return {"message": "Iris classifier is alive. POST to /predict with `point` as list of 4 numbers."}

@app.post("/predict")
def predict(input_data: PointInput):
    X = [input_data.point]  # single sample
    try:
        pred_idx = int(pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        probs = {name: float(p) for name, p in zip(target_names, proba)}
    else:
        probs = None

    return {"prediction": target_names[pred_idx], "probabilities": probs}
