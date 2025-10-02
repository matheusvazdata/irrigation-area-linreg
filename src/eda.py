import pandas as pd
from .config import REPORTS_FIGURES, FEATURE_COL, TARGET_COL
from .utils.plotting import save_scatter

def run_eda(df: pd.DataFrame) -> dict:
    summary = df.describe(include="all").to_dict()
    corr = df[[FEATURE_COL, TARGET_COL]].corr(method="pearson").iloc[0,1]
    # Scatter
    save_scatter(
        df[FEATURE_COL],
        df[TARGET_COL],
        xlabel=FEATURE_COL,
        ylabel=TARGET_COL,
        title=f"{FEATURE_COL} vs {TARGET_COL}",
        outpath=REPORTS_FIGURES / "scatter_hours_vs_area_per_angle.png",
    )
    return {"summary": summary, "pearson_corr_hours_area_per_angle": float(corr)}