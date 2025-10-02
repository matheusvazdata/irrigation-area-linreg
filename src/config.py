from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIAG = PROJECT_ROOT / "reports" / "diagnostics"

RAW_CSV = DATA_RAW / "irrigation_data.csv"

TARGET_COL = "Irrigated Area per Angle"
FEATURE_COL = "Irrigation Hours"

SEED = 42
TEST_SIZE = 0.2