"""Flask backend: serves the front end and exposes the REST API."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

import database as db
from common import FEATURES, LIMITS, MODEL_NAMES, clean_frame

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "models.joblib"

app = Flask(__name__, static_folder=str(BASE.parent / "frontend"), static_url_path="")
db.init_db()
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            return None
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_ready=MODEL_PATH.exists())


@app.post("/api/predict")
def predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify(error="Model not trained yet. Run: python train_model.py"), 503

    data = request.get_json(silent=True) or {}
    model_key = data.get("model", "random_forest")
    if model_key not in MODEL_NAMES:
        return jsonify(error="Unknown model."), 400

    values = {}
    for f in FEATURES:
        try:
            v = float(data.get(f))
        except (TypeError, ValueError):
            return jsonify(error=f"'{f}' is required and must be a number."), 400
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            return jsonify(error=f"'{f}' must be between {lo} and {hi}."), 400
        values[f] = v

    frame = clean_frame(pd.DataFrame([values], columns=FEATURES), bundle["medians"])
    model = bundle["models"][model_key]
    prob = float(model.predict_proba(frame)[0][1])
    pred = int(prob >= 0.5)

    db.add_prediction([values[f] for f in FEATURES], model_key, pred, prob)
    return jsonify(prediction=pred,
                   label="Diabetic (High Risk)" if pred else "Non-Diabetic (Low Risk)",
                   probability=round(prob, 4), model=MODEL_NAMES[model_key])


@app.get("/api/history")
def history():
    return jsonify(db.get_predictions(50))


@app.delete("/api/history")
def clear_history():
    db.clear_predictions()
    return jsonify(status="cleared")


@app.get("/api/metrics")
def metrics():
    rows = db.get_metrics()
    for r in rows:
        r["name"] = MODEL_NAMES.get(r["model"], r["model"])
    return jsonify(rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
