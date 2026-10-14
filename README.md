# Diabetes Risk Prediction (Flask + SQLite + HTML/CSS/JS)

```
diabetes-prediction/
├── backend/
│   ├── app.py            Flask server + REST API
│   ├── train_model.py    trains Logistic Regression & Random Forest (5-fold CV)
│   ├── database.py       SQLite layer (predictions + model_metrics tables)
│   ├── common.py         shared features, limits, preprocessing
│   ├── requirements.txt
│   └── data/diabetes.csv Pima Indians Diabetes Dataset
└── frontend/             index.html, style.css, script.js
```

## Steps in VS Code (Windows)

1. Install Python 3.10 - 3.12 from python.org (tick **Add Python to PATH**) and VS Code.
2. In VS Code install the **Python** extension (Microsoft).
3. **File > Open Folder...** and select `diabetes-prediction`.
4. Open a terminal: **Terminal > New Terminal**, then:
   ```
   cd backend
   python -m venv venv
   venv\Scripts\activate
   ```
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   (macOS/Linux: `python3 -m venv venv` and `source venv/bin/activate`).
5. Press **Ctrl+Shift+P > Python: Select Interpreter** and pick the one inside `backend/venv`.
6. Install libraries: `pip install -r requirements.txt`
7. Train the models (creates `models/models.joblib` and `diabetes.db`): `python train_model.py`
8. Start the server: `python app.py`
9. Open http://127.0.0.1:5000 in your browser. Click **Fill sample data**, then **Predict**.
10. Stop the server with **Ctrl+C**.

## API
| Method | URL | Purpose |
|---|---|---|
| POST | /api/predict | JSON with 8 features + `model` (`random_forest` or `logistic_regression`) |
| GET | /api/history | last 50 saved predictions |
| DELETE | /api/history | clear history |
| GET | /api/metrics | model evaluation results |
| GET | /api/health | server / model status |

## Troubleshooting
- `python` not found: reinstall Python with "Add to PATH", or use `py` instead of `python`.
- "Model not trained yet": run `python train_model.py` first (from inside `backend`).
- "Port 5000 in use": change `port=5000` at the bottom of `app.py`.
- `ModuleNotFoundError`: the venv is not active; repeat step 4 and 6.
- Blank metrics table: run `train_model.py` again.
