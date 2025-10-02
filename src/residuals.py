from pathlib import Path
import numpy as np
from scipy.stats import shapiro
import statsmodels.api as sm
import matplotlib.pyplot as plt
from .config import REPORTS_DIAG

def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = (y_true - y_pred).ravel()
    stat, pvalue = shapiro(residuals)
    # QQ-plot
    REPORTS_DIAG.mkdir(parents=True, exist_ok=True)
    fig = sm.qqplot(residuals, line="45")
    plt.title("Residuals QQ-plot")
    qq_path = REPORTS_DIAG / "residuals_qqplot.png"
    plt.savefig(qq_path, dpi=150)
    plt.close()

    return {
        "residuals_summary": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=1)),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        },
        "shapiro_stat": float(stat),
        "shapiro_pvalue": float(pvalue),
        "qqplot_path": str(qq_path)
    }