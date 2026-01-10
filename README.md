# Claxon

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A lightweight FastAPI model service that exposes a serialized DecisionTree model for inference, with basic data-drift checks against stored baseline distributions.

- Model artifact: decisiontree.sav
- Baseline distributions used for drift detection: baseline_distributions.pkl
- API entrypoint: mlapi.py
- Notebook (training / artifact generation): CLAXON.ipynb
- Procfile included for Heroku-style deployment

---

## Table of contents

- [Quick summary](#quick-summary)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quickstart — run locally](#quickstart---run-locally)
- [API — /predict](#api---predict)
  - [Request schema](#request-schema)
  - [Example request (curl)](#example-request-curl)
  - [Example responses](#example-responses)
- [Encodings & allowed values](#encodings--allowed-values)
- [Data drift checking](#data-drift-checking)
- [Regenerating model & baseline distributions](#regenerating-model--baseline-distributions)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

---

## Quick summary

Claxon provides:
- A REST endpoint to score individual loan-like applicants using a trained DecisionTree model.
- Simple pre-defined encodings for categorical features (gender, job, location, marital status).
- A lightweight drift detector that compares incoming feature values to baseline distributions and returns warnings if a drift is detected.

---

## Repository layout

Important files:
- `mlapi.py` — FastAPI application (loads model and baseline distributions)
- `decisiontree.sav` — pickled scikit-learn DecisionTreeClassifier used for inference
- `baseline_distributions.pkl` — baseline stats/dictionaries used for drift detection
- `CLAXON.ipynb` — Jupyter notebook (likely used to train the model and generate baseline distributions)
- `requirements.txt` — Python dependencies
- `Procfile` — command for running with Gunicorn (useful for Heroku)
- `runtime.txt` — pinned Python runtime: python-3.12.4

---

## Requirements

This project uses Python 3.12 (see `runtime.txt`). Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

(If you only want to run the API and avoid installing many development packages, consider creating a slim requirements file with the minimal runtime dependencies: fastapi, uvicorn, pandas, numpy, scikit-learn, joblib.)

---

## Quickstart — run locally

Run with Uvicorn (development):

```bash
uvicorn mlapi:app --host 0.0.0.0 --port 8000 --reload
```

Run with Gunicorn + Uvicorn worker (production-like; same command used in `Procfile`):

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker mlapi:app
```

The server will expose OpenAPI docs at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## API — /predict

POST /predict accepts JSON matching the ScoringItem model and returns a model prediction. If the incoming request shows distributional drift relative to `baseline_distributions.pkl`, the response will include a `drift_warning` field with details.

### Request schema

Send the following JSON fields:

- gender: str — one of `"female"`, `"male"`, `"other"` (unknown values handled with code -1)
- is_employed: int — 0 or 1
- job: str — job title (mapped internally)
- location: str — city name (mapped internally)
- loan_amount: float
- number_of_defaults: int
- outstanding_balance: float
- interest_rate: float
- age: int
- remaining_term: float — note: key in JSON is `remaining_term` (API maps it to internal column name `remaining term`)
- salary: float
- marital_status: str — one of `"single"`, `"married"`, `"divorced"` (unknown values handled with code -1)

Feature order expected by model (internal):  
gender, is_employed, job, location, loan_amount, number_of_defaults, outstanding_balance, interest_rate, age, remaining term, salary, marital_status

### Example request (curl)

```bash
curl -sS -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "is_employed": 1,
    "job": "Software developer",
    "location": "Harare",
    "loan_amount": 2500.0,
    "number_of_defaults": 0,
    "outstanding_balance": 100.0,
    "interest_rate": 8.5,
    "age": 30,
    "remaining_term": 24.0,
    "salary": 1200.0,
    "marital_status": "single"
  }'
```

### Example responses

- Normal prediction (no drift detected):

```json
{
  "prediction": 1
}
```

- Prediction with drift warning:

```json
{
  "prediction": 0,
  "drift_warning": [
    "Continuous feature 'salary' shows drift with Z-score: 2.34",
    "Categorical feature 'job' shows drift for category 'Unlisted Job'"
  ]
}
```

Notes:
- `prediction` is returned as an integer (converted from model output).
- `drift_warning` is an array of human-readable strings describing features that exceeded thresholds.

---

## Encodings & allowed values

mlapi.py applies simple encodings and mappings before prediction:

- Gender recoding:
  - "female" -> 0
  - "male" -> 1
  - "other" -> 2
  - unknown -> -1

- Marital status recoding:
  - "single" -> 0
  - "married" -> 1
  - "divorced" -> 2
  - unknown -> -1

- Job mapping (example):
  - 'Teacher' -> 8
  - 'Nurse' -> 6
  - 'Doctor' -> 3
  - 'Data analyst' -> 1
  - 'Software developer' -> 7
  - 'Accountant' -> 0
  - 'Lawyer' -> 5
  - 'Engineer' -> 4
  - 'Data scientist' -> 2
  - unknown -> -1

- Location mapping: (partial list — keep the mapping in mlapi.py)
  - 'Beitbridge': 0, 'Bulawayo': 1, 'Chimanimani': 2, 'Chipinge': 3, ..., 'Victoria falls': 22, 'Zvishavane': 23
  - unknown -> -1

If you add new jobs or locations, update `job_mapping` and `location_mapping` in `mlapi.py`.

---

## Data drift checking

Before returning a final prediction, the API performs a simple drift check using the loaded `baseline_distributions.pkl`. Behavior:

- Continuous features: compares the incoming mean to baseline mean, computes a Z-score using baseline std. If Z-score > threshold (default threshold = 0.05 in code), a drift is flagged.
- Categorical features: compares category proportions to baseline proportions; if difference > threshold a drift is flagged. Also flags when new categories are present.
- If any drift is detected, the API:
  - Logs drift details (to stdout / server logs)
  - Returns prediction together with `drift_warning` list (strings describing drift)

Note: The threshold values and detection logic are basic and intended as a lightweight check. Adjust in `mlapi.py` (function `check_data_drift`) if you need different sensitivity.

---

## Regenerating model & baseline distributions

- The repository includes `CLAXON.ipynb`. That notebook likely contains the data preparation, model training, and code used to generate:
  - `decisiontree.sav` (pickled model)
  - `baseline_distributions.pkl` (baseline statistics/dictionaries)

To regenerate artifacts:
1. Open the notebook in Jupyter:
   - `jupyter notebook CLAXON.ipynb` or `jupyter lab`
2. Re-run the training cells and update paths to save `decisiontree.sav` and `baseline_distributions.pkl`.
3. Replace the files in the repository or update your deployment storage.

When updating the model, remember to keep the same expected feature order and encoding, or update `mlapi.py` accordingly.

---

## Deployment

This repository contains a `Procfile` configured for Heroku-style deployments:

Procfile:
```
web: gunicorn -w 2 -k uvicorn.workers.UvicornWorker mlapi:app
```

runtime.txt pins Python to 3.12.4. On Heroku:
- Ensure the `decisiontree.sav` and `baseline_distributions.pkl` files are included in the slug (or loaded from remote storage).
- Deploy normally (git push heroku main).

Alternatively, deploy to any containerized platform; a simple Dockerfile can run the same gunicorn command. If you need help creating a Dockerfile, open an issue or contribute one.

---

## Troubleshooting

- "Module not found" errors: ensure you installed dependencies from `requirements.txt` into the environment used to run the server.
- Model load error: Confirm `decisiontree.sav` exists and is compatible with the scikit-learn version in `requirements.txt`.
- Unexpected prediction values or errors about missing columns: Ensure the request JSON uses the exact keys listed in the Request schema; `remaining_term` must be provided (it is mapped to `remaining term` internally).
- If you get `-1` encodings for job/location, either pass a supported name or update the mapping in `mlapi.py`.

---

## License

This project is provided under the MIT License.

You can include a top-level LICENSE file with the same text below (recommended). Replace [year] and [copyright owner] with appropriate values.

```
MIT License

Copyright (c) [2024] [copyright Blessed Zhou]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```



## Contributing

Contributions are welcome. Suggested changes:
- Add unit/integration tests around mlapi (validate encoding, model loading, and drift behavior).
- Add a minimal `requirements-min.txt` for deployments.
- Improve drift detection (configurable thresholds, statistical tests).
- Add CI and a Dockerfile for containerized deployment.

To contribute:
1. Fork the repository
2. Create a branch for your feature/fix
3. Open a pull request with a clear description of the change

---

If you have questions or want help extending the API (adding batch inference, authentication, or improved drift detection), open an issue or submit a PR.
