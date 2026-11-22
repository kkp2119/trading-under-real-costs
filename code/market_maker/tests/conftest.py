"""Research 12 engine tests: synthetic data only (no bar file is opened). Files go under $R12_TEST_DIR (outside the
repository, nothing deleted). Run with python -m pytest -p no:cacheprovider research12/tests."""
import os
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]


@pytest.fixture
def tmpdir_r12():
    base = os.environ.get("R12_TEST_DIR")
    if not base:
        pytest.skip("set R12_TEST_DIR (outside the repository)")
    d = os.path.join(base, uuid.uuid4().hex[:10])
    os.makedirs(d)
    return d
