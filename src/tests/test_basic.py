import json
from pathlib import Path

def test_metadata_exists():
    meta = Path("artifacts/metadata.json")
    assert meta.exists(), "Train the model first to generate metadata.json"
    data = json.loads(meta.read_text())
    assert "equation" in data