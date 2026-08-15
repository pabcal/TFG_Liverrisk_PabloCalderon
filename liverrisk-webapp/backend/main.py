"""
LiverRisk webapp backend.

This is a small FastAPI server with a single job: take a CSV upload
describing one or a few patients (same raw columns as train.csv /
test.csv), run them through the already-trained models in models/, and
return a risk score for each patient.

Startup loading and shared scoring helpers live in models_loader.py.
The routes are split into their own modules: predict.py,
sample_patients.py, rankings.py, disagreement.py, about.py.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from predict import router as predict_router
from sample_patients import router as sample_patients_router
from rankings import router as rankings_router
from disagreement import router as disagreement_router
from about import router as about_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="LiverRisk webapp")

app.include_router(predict_router)
app.include_router(sample_patients_router)
app.include_router(rankings_router)
app.include_router(disagreement_router)
app.include_router(about_router)


# --------------------------------------------------------------------
# Serve the plain HTML/CSS/JS frontend from the same server, so the
# page can call /predict on its own origin with a plain fetch() call.
# This must be mounted last -- routes defined above (like /predict)
# are matched first, and anything else falls through to these static
# files.
# --------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
