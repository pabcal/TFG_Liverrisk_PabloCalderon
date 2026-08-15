"""
LiverRisk webapp backend -- GET /about route.

Backs the "Behind the Scenes" tab. Everything it returns was already
computed by notebooks/04_results_analysis.ipynb and written to
precomputed/about_stats.json -- this route just reads that file back.
No computation happens here or on the frontend; the four charts it
references (feature importance x2, Kaplan-Meier x2) are plain PNGs the
same notebook saved to frontend/images/, served as static files.
"""
from __future__ import annotations

import json

from fastapi import APIRouter

from models_loader import REPO_ROOT

router = APIRouter()

ABOUT_STATS_PATH = REPO_ROOT / "liverrisk-webapp" / "backend" / "precomputed" / "about_stats.json"


@router.get("/about")
async def about():
    with open(ABOUT_STATS_PATH) as f:
        return json.load(f)
