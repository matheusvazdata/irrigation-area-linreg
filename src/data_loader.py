import pandas as pd
from .config import RAW_CSV

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # Normalize/rename columns to English
    rename_map = {
        "Horas de Irrigação": "Irrigation Hours",
        "Área Irrigada": "Irrigated Area",
        "Área Irrigada por Ângulo": "Irrigated Area per Angle",
    }
    # Fall back if already English
    for k, v in list(rename_map.items()):
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    expected = {"Irrigation Hours", "Irrigated Area", "Irrigated Area per Angle"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df