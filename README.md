# Predictive Subscription Model

A FastAPI web API that predicts whether a client will subscribe (yes/no) to a service (e.g., bank term deposit) based on client and campaign features. Built on the UCI Bank Marketing dataset.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Model Training (Notebook)](#model-training-notebook)
5. [API Usage](#api-usage)
6. [Installation & Running](#installation--running)
7. [Example Request](#example-request)
8. [Known Issues / Limitations](#known-issues--limitations)

## Overview

Predicts subscription outcome (`yes`/`no`) via a trained Logistic Regression model, served as a REST API with FastAPI. The API accepts 13 encoded features and returns a prediction label.

## Project Structure

```
Predictive-Subscription-Model/
├── pred.py                     # FastAPI inference API
├── best_model.pkl              # Serialized trained model
├── data.csv                    # Training dataset (8,237 rows)
├── Subscription_Processing.ipynb  # EDA + training notebook
├── ScreenShot.pdf              # Project documentation/screenshots
├── .venv/                      # Python virtual environment
├── .gitignore
└── README.md
```

## Dataset (`data.csv`)

Source: Bank Marketing dataset (UCI)

- 8,237 rows × 17 columns
- Target column: `y` (yes/no)
- Features used by the model (after dropping `Unnamed: 0`, `month`, `day_of_week`):
  - Numeric: `age`, `duration`, `campaign`, `pdays`, `previous`
  - Categorical (label-encoded): `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `poutcome`

## Model Training (`Subscription_Processing.ipynb`)

- Loads and cleans data (drops ID/date columns, fills missing `duration`/`job`, removes duplicates)
- Label-encodes 8 categorical columns
- Splits data 80/20 train/test (`random_state=0`)
- Applies `StandardScaler` to numeric features
- Compares models on the test set:

| Model | Accuracy | Yes-class recall |
|-------|----------|------------------|
| Random Forest | 0.90 | 0.35 |
| Logistic Regression | 0.91 | 0.31 |

- Logistic Regression selected as the best model
- Saved to `best_model.pkl` via `pickle.dump`

## API Usage (`pred.py`)

- `GET /` → info message
- `POST /predict` → JSON body with 13 features, returns `{"prediction": <label>}`

## Installation & Running

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install fastapi uvicorn joblib scikit-learn pandas pydantic numpy

# Run the server
uvicorn pred:app --reload

# Interactive API docs
http://127.0.0.1:8000/docs
```

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "job": 2,
    "marital": 1,
    "education": 3,
    "default": 0,
    "housing": 1,
    "loan": 0,
    "contact": 1,
    "duration": 200.0,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": 2
  }'
```

> Categorical values must be pre-encoded as integers matching the notebook's `LabelEncoder` mapping.

## Known Issues / Limitations

1. **Serialization mismatch:** `pred.py` loads the model with `joblib.load()`, but the notebook saves with `pickle.dump()`. This may fail depending on library versions.
2. **Missing scaling in API:** The model was trained on `StandardScaler`-scaled numeric features, but `pred.py` sends raw unscaled values directly to `predict()`. This can cause incorrect predictions.
3. **No input encoding in API:** Callers must manually label-encode categorical fields before sending.
