import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurance_ml.data import load_raw_data  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "insuranceFINAL.csv"


@pytest.fixture(scope="session")
def df_raw():
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not present at {DATA_PATH} — place insuranceFINAL.csv there to run data-dependent tests")
    return load_raw_data(DATA_PATH)
