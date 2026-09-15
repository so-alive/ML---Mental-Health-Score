# 🧠 Mental Health Score Predictor

A full-stack machine learning system that predicts a student's mental well-being score from their social media habits, sleep, study load, physical activity, and stress level — served through a live FastAPI backend and a simple web frontend.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)

🔗 **Live App:** [ml-mental-health-score-1.onrender.com](https://ml-mental-health-score-1.onrender.com)
*(hosted on Render's free tier — the backend spins down when idle, so the first request after inactivity can take 30–50s to wake up.)*

---

> ⚠️ **Disclaimer:** This project is for educational and portfolio purposes only. The predicted score is not a clinical diagnosis, psychological assessment, or substitute for professional mental health support. If you or someone you know is struggling, please reach out to a qualified professional or a trusted person.

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [API Documentation](#-api-documentation)
- [Model & Training Pipeline](#-model--training-pipeline)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)

---

## 🧩 Overview

Most student wellness tools are either a static self-report survey or a black-box score with no visible reasoning. This project takes a different approach: it trains a regression model on real behavioral and lifestyle data (screen time, sleep, study hours, stress, etc.), wraps it in a validated REST API, and exposes it through a lightweight web form — so a user gets an instant, model-backed estimate of their mental health score along with the inputs that drove it.

The project covers the full ML lifecycle:

1. **Exploratory Data Analysis** — distributions, correlations, outlier detection
2. **Data Cleaning & Feature Engineering** — deduplication, outlier clipping, skew correction, country grouping
3. **Model Building** — baseline Linear Regression vs. Random Forest, with hyperparameter tuning
4. **Model Evaluation** — R², MAE, RMSE comparison across models
5. **Deployment** — a FastAPI service serving the trained pipeline, with a simple frontend for end users

Whether you're exploring ML pipelines end-to-end or just want to check in on your own habits, the app gives you an instant, transparent prediction.

## ✨ Key Features

### 📊 1. Exploratory Data Analysis
- Target distribution with KDE overlay (`Mental_Health_Score`)
- Correlation heatmap across numeric features
- Stress-level boxplots, usage/sleep/activity scatterplots against the target
- IQR-based outlier detection across all numeric columns

### 🧹 2. Data Cleaning & Feature Engineering
- Duplicate row removal
- Negative/unrealistic value clipping (e.g. `Physical_Activity_Hours`)
- Skewness correction on `Study_Hours` via log-transform
- Smart country grouping — 111 raw countries collapsed into the top 10 + "Other" to keep one-hot encoding manageable

### ⚙️ 3. Unified Preprocessing Pipeline
- A single `ColumnTransformer` handling skewed, numeric, ordinal, and nominal features identically at train and inference time
- Ordinal encoding preserves the natural Low → Medium → High → Very High stress order (instead of losing it to one-hot encoding)

### 🌲 4. Model Building & Evaluation
- Linear Regression baseline
- Random Forest (default and hyperparameter-tuned via `RandomizedSearchCV`)
- R² / MAE / RMSE comparison across all three, with the best-performing model selected and serialized

### ⚡ 5. Production Prediction API
- FastAPI service with a validated `/predict` endpoint
- Pydantic request/response schemas with field-level constraints (age ranges, allowed platforms, stress levels)
- CORS-enabled, callable from any frontend

### 📱 6. Web Frontend
- A plain HTML/CSS/JS form that posts to the API and renders the predicted score
- No framework required — vanilla `fetch()` handles the whole integration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Web Frontend                          │
│              (HTML, CSS, vanilla JavaScript)                │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  │  POST /predict (JSON)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI                              │
│                        main.py                              │
│           Pydantic request/response validation              │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  │  pandas DataFrame
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Trained scikit-learn Pipeline                  │
│               Mental_Health_Model.pkl                       │
│        (ColumnTransformer + RandomForestRegressor)          │
└─────────────────┬─────────────────────────────────────────┘
                  ▼
        Predicted Mental Health Score (0–10)
```

## 🧰 Tech Stack

**Frontend**

| Technology | Purpose |
|---|---|
| HTML / CSS | Structure and styling |
| JavaScript (`fetch`) | API calls and dynamic score rendering |

**Backend**

| Technology | Purpose |
|---|---|
| FastAPI | REST API server |
| Pydantic | Request/response schema validation |
| Uvicorn | ASGI server |
| CORS Middleware | Cross-origin support for the frontend |

**Machine Learning**

| Technology | Purpose |
|---|---|
| scikit-learn `1.6.1` | Preprocessing pipeline + Random Forest model |
| pandas / NumPy | Data wrangling, inference-time transforms |
| joblib | Model serialization / deserialization |
| matplotlib / seaborn | Notebook-side EDA and visualization |

**Hosting**

| Service | Purpose |
|---|---|
| Render | Backend deployment (`runtime.txt` pins Python 3.12.10) |

## 📊 Dataset

**Source:** `Student Social Media And Mental Health Impact.csv` — 5,000 student records, 13 columns, no missing values.

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Student age |
| `Gender` | Categorical | Male / Female |
| `Country` | Categorical | Raw country (grouped into top 10 + "Other" for modeling) |
| `Academic_Level` | Categorical | High School / Undergraduate / Graduate |
| `Most_Used_Platform` | Categorical | Primary social media platform |
| `Purpose_Of_Use` | Categorical | Networking / Education / Entertainment / News |
| `Avg_Daily_Usage_Hours` | Numeric | Average daily screen time |
| `Daily_Unlocks` | Numeric | Average daily phone unlocks |
| `Study_Hours` | Numeric (skewed) | Daily study hours |
| `Physical_Activity_Hours` | Numeric | Daily physical activity |
| `Sleep_Hours_Per_Night` | Numeric | Average nightly sleep |
| `Stress_Level` | Ordinal | Low → Medium → High → Very High |
| `Mental_Health_Score` | **Target** | Continuous score, ~3.6 to 9.4 |

## 🚀 Getting Started

**Prerequisites**
- Python 3.12+
- pip

**Installation**

1. Clone Repository
```bash
git clone https://github.com/so-alive/ML---Mental-Health-Score.git
cd ML---Mental-Health-Score
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the API
```bash
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`, with interactive docs auto-generated at `http://127.0.0.1:8000/docs`.

4. Run the Frontend

Open `index.html` directly, or serve it with a static server. Make sure `API_BASE` in `script.js` points to wherever the backend is running (local or the deployed Render URL).

## 📡 API Documentation

### Health Check
`GET /`

**Response**
```json
{ "hope you're having a great day" }
```

### Prediction Endpoint
`POST /predict`

Predicts a mental health score from student lifestyle data.

**Request body**
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 150,
  "study_hours": 3.0,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.0,
  "stress_level": "Medium"
}
```

**Field constraints**

| Field | Constraint |
|---|---|
| `age` | integer, 10–100 |
| `gender` | `Male` \| `Female` |
| `academic_level` | `Undergraduate` \| `Graduate` \| `High School` |
| `most_used_platform` | Facebook, LinkedIn, Instagram, Snapchat, Twitter, YouTube, TikTok, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| `purpose_of_use` | `Networking` \| `Education` \| `Entertainment` \| `News` |
| `avg_daily_usage_hours` | float, 0–24 |
| `daily_unlocks` | integer, ≥ 0 |
| `study_hours` | float, 0–24 |
| `physical_activity_hours` | float, 0–2 |
| `sleep_hours_per_night` | float, 0–24 |
| `stress_level` | `Low` \| `Medium` \| `High` \| `Very High` |

**Response**
```json
{ "predicted_mental_health_score": 6.82 }
```

Countries outside the training data's top 10 (India, USA, Canada, Australia, UK, Germany, Mexico, Turkey, France) are automatically grouped into `"Other"` before prediction, matching how the model was trained.

## 🤖 Model & Training Pipeline

### Preprocessing Pipeline (`ColumnTransformer`)

| Feature group | Columns | Transform |
|---|---|---|
| Skewed numeric | `Study_Hours` | `log1p` → `StandardScaler` |
| Plain numeric | `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` | `StandardScaler` |
| Ordinal | `Stress_Level` | `OrdinalEncoder` (Low < Medium < High < Very High) |
| Nominal | `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country` | `OneHotEncoder` |

### Model Comparison (30% held-out test set)

| Model | Test R² | Train R² | MAE | RMSE |
|---|:---:|:---:|:---:|:---:|
| Linear Regression | 0.7398 | 0.7237 | 0.536 | 0.676 |
| **Random Forest (default)** ✅ | **0.8776** | 0.9808 | **0.347** | **0.464** |
| Random Forest (tuned) | 0.8650 | 0.9547 | 0.369 | 0.487 |

The **default Random Forest** was selected as the final model — it outperformed the hyperparameter-tuned version on held-out test data, despite the tuned model showing a smaller train/test gap. It's saved as `Mental_Health_Model.pkl` via `joblib` and loaded directly by the FastAPI service at startup.

<details>
<summary>Hyperparameter search space</summary>

```python
param_grid = {
    'random forest__n_estimators': [100, 200, 300],
    'random forest__max_depth': [5, 10, 15],
    'random forest__min_samples_split': [2, 5, 10],
    'random forest__min_samples_leaf': [1, 2, 4]
}
# RandomizedSearchCV: n_iter=15, cv=5, scoring='r2'
```

</details>

## 📁 Project Structure

```
ML---Mental-Health-Score/
├── ML_Mental_Health_Score.ipynb                      # EDA, cleaning, feature engineering, training & evaluation
├── Mental_Health_Model.pkl                           # Serialized sklearn pipeline (preprocessing + Random Forest)
├── Student Social Media And Mental Health Impact.csv # Source dataset
├── main.py                                           # FastAPI app and /predict endpoint
├── index.html                                        # Frontend form
├── style.css                                         # Frontend styling
├── script.js                                         # Frontend logic / API calls
├── requirements.txt                                  # Backend dependencies
├── runtime.txt                                       # Pinned Python version for deployment
└── README.md
```

## 🚢 Deployment

Deployed on **Render**, using `requirements.txt` for dependencies and `runtime.txt` (`python-3.12.10`) to pin the runtime.

```bash
# Build command
pip install -r requirements.txt

# Start command
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Running on Render's free tier, the service spins down after periods of inactivity — expect a 30–50 second cold start on the first request.
