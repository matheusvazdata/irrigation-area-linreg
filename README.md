# Irrigation Linear Regression Project

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Uvicorn-4B8BBE?style=for-the-badge&logo=uvicorn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

This project implements a **complete machine learning workflow** to analyze irrigation data, train a **Linear Regression model**, evaluate its performance, run residual diagnostics, and serve predictions through a **FastAPI web API**.

The goal is to predict the **Irrigated Area per Angle** based on the number of **Irrigation Hours**, enabling data-driven decision-making in agricultural water management.

## 📂 Project Structure

```
irrigation-area-linreg/
├── artifacts/                        # Trained model and metadata
│   ├── metadata.json
│   └── model.joblib
├── data/
│   ├── processed/                    # Intermediate/processed data (ignored in .gitignore)
│   └── raw/                          # Raw input data
│       └── irrigation_data.csv
├── reports/                          # Generated plots and diagnostics
│   ├── diagnostics/
│   │   └── residuals_qqplot.png
│   └── figures/
│       ├── scatter_hours_vs_area_per_angle.png
│       └── true_vs_pred.png
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # End-to-end pipeline script
└── src/                              # Source code
    ├── api/
    │   └── main.py                   # FastAPI application
    ├── config.py                     # Project configuration (paths, constants)
    ├── data_loader.py                # Load and preprocess data
    ├── eda.py                        # Exploratory Data Analysis
    ├── evaluator.py                  # Model evaluation (MSE, MAE)
    ├── residuals.py                  # Residual analysis and normality checks
    ├── tests/
    │   └── test_basic.py             # Basic unit test
    ├── trainer.py                    # Model training and persistence
    └── utils/
        └── plotting.py               # Plotting utilities
```

## ⚙️ Installation & Setup

### 1. Clone the repository

You can clone the repository either via **HTTPS** or **SSH**:

**HTTPS (recommended for beginners):**
```bash
git clone https://github.com/matheusvazdata/irrigation-area-linreg.git
cd irrigation-area-linreg
```

**SSH (recommended if you already configured your SSH keys with GitHub):**

```bash
git clone git@github.com:matheusvazdata/irrigation-area-linreg.git
cd irrigation-area-linreg
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows (PowerShell)
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Data

This project already includes the dataset inside the repository, located at:

```
data/raw/irrigation_data.csv
```

Columns:

* `Irrigation Hours` – Number of hours of irrigation
* `Irrigated Area` – Total irrigated area
* `Irrigated Area per Angle` – Irrigated area distributed per irrigation angle

> No manual download is required — the pipeline will read this file directly.

## 🚀 Running the Pipeline

To execute the entire pipeline (EDA → Training → Evaluation → Residuals):

```bash
python run_pipeline.py
```

This will:

* Generate plots in `reports/figures/`
* Train the linear regression model and save it to `artifacts/model.joblib`
* Save model metadata in `artifacts/metadata.json`
* Print evaluation metrics (MSE, MAE)
* Run residual diagnostics and save a QQ-plot in `reports/diagnostics/`

## 📈 Generated Reports

* **Scatter plot**: `reports/figures/scatter_hours_vs_area_per_angle.png`
* **True vs Predicted**: `reports/figures/true_vs_pred.png`
* **Residuals QQ-plot**: `reports/diagnostics/residuals_qqplot.png`

These plots allow you to visually inspect linearity, model fit, and residual normality.

## 🤖 Model Serving (API)

The project includes a **FastAPI** server that exposes endpoints for real-time predictions.

### Run the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

#### Healthcheck

```http
GET /health
```

#### Model Info

```http
GET /model-info
```

Returns metadata about the trained model (equation, coefficients, test size, etc).

#### Predict (single input)

```http
POST /predict
Content-Type: application/json
{
  "irrigation_hours": 15
}
```

Response:

```json
{
  "irrigated_area_per_angle": 1000.0
}
```

#### Predict Batch (multiple inputs)

```http
POST /predict-batch
Content-Type: application/json
{
  "irrigation_hours": [1, 2, 5, 10, 15]
}
```

Response:

```json
{
  "irrigated_area_per_angle": [66.67, 133.33, 333.33, 666.67, 1000.0]
}
```

## 🧪 Testing

To run the basic unit test:

```bash
pytest src/tests -q
```

## 🔍 Key Features

* **Reproducible ML pipeline**: Data loading, EDA, training, evaluation, diagnostics
* **Persistence**: Model saved with `joblib` and metadata in JSON
* **Reports**: Plots automatically generated for interpretability
* **API ready**: Serve predictions with FastAPI
* **Extensible**: Easy to adapt for new datasets or models

## 🛠️ Possible Improvements

* Add noise injection for more realistic datasets
* Implement polynomial regression or regularization (Ridge, Lasso)
* Add CI/CD integration (GitHub Actions)
* Containerize with Docker for deployment

## 👤 Author

Developed by **Francisco Matheus Vaz dos Santos**

* GitHub: [github.com/matheusvazdata](https://github.com/matheusvazdata)
* LinkedIn: [linkedin.com/in/matheusvazdata](https://www.linkedin.com/in/matheusvazdata)
* Datacamp: [datacamp.com/portfolio/matheusvazdata](https://www.datacamp.com/portfolio/matheusvazdata)