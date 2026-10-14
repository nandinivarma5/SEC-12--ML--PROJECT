"""SQLite database layer (no external server needed)."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "diabetes.db"


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            pregnancies REAL, glucose REAL, blood_pressure REAL,
            skin_thickness REAL, insulin REAL, bmi REAL,
            pedigree REAL, age REAL,
            model TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_metrics (
            model TEXT PRIMARY KEY,
            accuracy REAL, precision REAL, recall REAL, f1 REAL,
            cv_accuracy REAL, tn INTEGER, fp INTEGER, fn INTEGER, tp INTEGER,
            trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)


def add_prediction(values, model, prediction, probability):
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO predictions (pregnancies, glucose, blood_pressure,
               skin_thickness, insulin, bmi, pedigree, age, model, prediction, probability)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (*values, model, prediction, probability))
        return cur.lastrowid


def get_predictions(limit=50):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def clear_predictions():
    with connect() as conn:
        conn.execute("DELETE FROM predictions")


def save_metrics(rows):
    with connect() as conn:
        for r in rows:
            conn.execute(
                """INSERT OR REPLACE INTO model_metrics
                   (model, accuracy, precision, recall, f1, cv_accuracy, tn, fp, fn, tp, trained_at)
                   VALUES (:model,:accuracy,:precision,:recall,:f1,:cv_accuracy,:tn,:fp,:fn,:tp,CURRENT_TIMESTAMP)""", r)


def get_metrics():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM model_metrics ORDER BY model").fetchall()
    return [dict(r) for r in rows]
