import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from .config import (
    FEATURE_COL, TARGET_COL, TEST_SIZE, SEED, ARTIFACTS_DIR, REPORTS_FIGURES
)
from .utils.plotting import save_line

def train_linear_regression(df: pd.DataFrame) -> dict:
    X = df[[FEATURE_COL]].values
    y = df[[TARGET_COL]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    coef = float(model.coef_.ravel()[0])
    intercept = float(model.intercept_.ravel()[0])
    eq = f"y = {coef:.6f} * x + {intercept:.6f}"

    y_pred = model.predict(X_test).ravel()

    # Persist artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metadata = {
        "model_type": "LinearRegression",
        "equation": eq,
        "coef": coef,
        "intercept": intercept,
        "feature": FEATURE_COL,
        "target": TARGET_COL,
        "random_state": SEED,
        "test_size": TEST_SIZE,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Quick line plot of predictions vs truth (sorted by index of test split)
    save_line(
        y_true=y_test.ravel(),
        y_pred=y_pred,
        title="Test Set: True vs Predicted",
        outpath=REPORTS_FIGURES / "true_vs_pred.png",
    )

    return {
        "equation": eq,
        "coef": coef,
        "intercept": intercept,
        "X_test": X_test.ravel().tolist(),
        "y_test": y_test.ravel().tolist(),
        "y_pred": y_pred.tolist(),
    }