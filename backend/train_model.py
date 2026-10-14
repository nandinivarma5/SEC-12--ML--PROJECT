"""Train Logistic Regression and Random Forest, evaluate, save model + metrics."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import database as db
from common import FEATURES, ZERO_AS_MISSING, clean_frame

BASE = Path(__file__).parent
DATA = BASE / "data" / "diabetes.csv"
MODEL_PATH = BASE / "models" / "models.joblib"


def main():
    df = pd.read_csv(DATA)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X, y = df[FEATURES], df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # medians computed on the training data only (no leakage)
    medians = {c: float(X_train[c].replace(0, float("nan")).median()) for c in ZERO_AS_MISSING}
    X_train, X_test = clean_frame(X_train, medians), clean_frame(X_test, medians)

    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        row = dict(model=name, accuracy=accuracy_score(y_test, pred),
                   precision=precision_score(y_test, pred), recall=recall_score(y_test, pred),
                   f1=f1_score(y_test, pred), cv_accuracy=cv_acc,
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        rows.append(row)
        print(f"{name:20s} acc={row['accuracy']:.3f} prec={row['precision']:.3f} "
              f"rec={row['recall']:.3f} f1={row['f1']:.3f} cv={cv_acc:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"models": models, "medians": medians}, MODEL_PATH)
    db.init_db()
    db.save_metrics(rows)
    print(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {db.DB_PATH}")


if __name__ == "__main__":
    main()
