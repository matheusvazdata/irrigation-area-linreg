import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_regression(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"MSE": mse, "MAE": mae}