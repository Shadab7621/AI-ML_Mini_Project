import sys
import warnings
from pathlib import Path

import pytest

# make `import backend` work when pytest is started from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from backend import datasets  # noqa: E402


@pytest.fixture()
def uploads(tmp_path, monkeypatch):
    """Uploaded files go to a temporary folder, never into the project."""
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path)
    return tmp_path


def csv_bytes(text: str) -> bytes:
    return text.strip().replace("\r\n", "\n").encode("utf-8")
