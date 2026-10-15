import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # for `import synth`


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real user config dir during tests."""
    monkeypatch.setenv("ORBIT_HOME", str(tmp_path / "orbit_home"))
    yield
