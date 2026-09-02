# LiverRisk

LiverRisk is a machine learning tool that predicts the risk of a major hepatic event and death in patients with MASLD. Instead of looking at a single lab result, it looks at a patient's full history over time. This lets it pick up on things a one time snapshot simply cannot see, like whether a patient's condition is actively getting worse.

This project was built for the ANNITIA challenge, an international competition to build better MASLD risk models. It was later extended into a full bachelor's thesis, including a rigorous comparison against the clinical formulas doctors already use, and a working web application doctors can actually try.

## What it actually does

Three different survival models are trained on a patient's longitudinal record. A penalized Cox model, a Random Survival Forest, and a gradient boosted model. These three are then blended together into one final ensemble.

The ensemble was tested against FIB-4 and APRI, the two formulas most commonly used today. On the official weighted score, LiverRisk reached 0.849. FIB-4 reached 0.768. APRI reached 0.671. In plain terms, LiverRisk is able to correctly tell which of two patients is at higher risk about 85 times out of 100. FIB-4 gets this right about 77 times out of 100.

## Quick start

If you just want to see the app running, this is all you need. The repository already includes the trained models and the data, so nothing needs to be built or retrained first.

```
git clone https://github.com/pabcal/TFG_Liverrisk_PabloCalderon.git
cd TFG_Liverrisk_PabloCalderon
uv sync
cd liverrisk-webapp/backend
uv run uvicorn main:app --reload
```

Then open the address shown in your terminal in a browser.

Full setup instructions, including how to regenerate everything from scratch, are in the thesis itself, Appendix A.

## What's in this repository

- `liverrisk/` The core Python package. This is where the actual logic lives, feature engineering, model definitions, cross validation, and blending.
- `notebooks/` The full pipeline, from raw data to final trained models, in order.
- `models/` The trained models, already fitted and ready to use.
- `liverrisk-webapp/` The web application, backend and frontend together.
- `exploratory_work/` Early exploration and testing from before the final approach was chosen. Kept here for transparency, not meant to represent the final methodology.

## Why this approach

Most of what makes this project different comes down to one idea. A patient is not a single number measured once. A patient's condition moves over time, and that movement carries real information. Formulas like FIB-4 cannot see that movement, since they only ever look at one visit at a time. This project was built specifically to capture it.
