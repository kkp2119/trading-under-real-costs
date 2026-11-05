"""Research 14 shell tests: synthetic data only (research13/tests/synth13.py). Files go under $R13_TEST_DIR (outside
the repository, nothing deleted). Run with:
    R13_TEST_DIR=<scratch> python -m pytest -p no:cacheprovider research13/study14/tests"""
import os
import sys
import uuid

import pytest

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R13 = os.path.dirname(STUDY)
sys.path[:0] = [STUDY, R13, os.path.join(R13, "tests")]


@pytest.fixture
def tmp14():
    base = os.environ.get("R13_TEST_DIR")
    if not base:
        pytest.skip("set R13_TEST_DIR (outside the repository)")
    d = os.path.join(base, "s14_" + uuid.uuid4().hex[:10])
    os.makedirs(d)
    return d
