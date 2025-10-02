from src.data_loader import load_raw_data
from src.eda import run_eda
from src.trainer import train_linear_regression
from src.evaluator import evaluate_regression
from src.residuals import residual_diagnostics
import numpy as np

def main():
    df = load_raw_data()
    eda_out = run_eda(df)
    print("[EDA] Pearson corr (Hours vs Area/Angle):", eda_out["pearson_corr_hours_area_per_angle"])

    train_out = train_linear_regression(df)
    print("[MODEL] Equation:", train_out["equation"])

    y_true = np.array(train_out["y_test"])
    y_pred = np.array(train_out["y_pred"])
    eval_out = evaluate_regression(y_true, y_pred)
    print("[EVAL]", eval_out)

    resid_out = residual_diagnostics(y_true, y_pred)
    print("[RESIDUALS]", resid_out)

if __name__ == "__main__":
    main()