import os
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

# --- EXPECTED CONSTANTS FROM DATASHEET ---

# Expected columns and their definitions
EXPECTED_COLUMNS = [
    "dataset",
    "boundary",
    "strategy",
    "detector",
    "seed",
    "error_stream",
    "trues",
    "preds",
    "max_delay",
    "max_early",
]

# Shared boundary types
BOUNDARIES = ["abrupt", "gradual", "slow"]

# CLDD-A specific configurations
DATASETS_A = ["rfmnist", "core50"]
STREAM_LENGTHS = {"rfmnist": 300_000, "core50": 126_500}
STRATEGIES_A = ["EWC", "FT", "LWF", "MAS", "RWalk", "SI"]
DETECTORS_A = ["oracle"]

# CLDD-B specific configurations
STRATEGIES_B = ["EWC", "LWF", "MAS", "RWalk", "SI"]
DETECTORS_B = ["ADWIN", "DDM", "PH", "SEED", "STEPD"]

# --- FIXTURES ---


@pytest.fixture(scope="module")
def cldd_a():
    """Loads CLDD-A from parquet format[cite: 1]."""
    return pd.read_parquet(Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "CLDD_A.parquet")


@pytest.fixture(scope="module")
def cldd_b():
    """Loads CLDD-B from parquet format[cite: 1]."""
    return pd.read_parquet(Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "CLDD_B.parquet")


# --- TESTS FOR CLDD-A ---


def test_cldd_a_columns(cldd_a):
    """Test that CLDD-A contains all documented columns[cite: 1]."""
    assert list(cldd_a.columns) == EXPECTED_COLUMNS, (
        "CLDD-A columns do not match the datasheet."
    )


def test_cldd_a_factorial_size(cldd_a):
    """
    CLDD-A is a full factorial of 3 boundaries × 6 strategies × 1 detector × 20 seeds.
    Total count should be 180[cite: 1].
    """
    assert len(cldd_a) == 180 * 2 * len(DATASETS_A), (
        "CLDD-A should have exactly 360 configurations per dataset."
    )


def test_cldd_a_unique_values(cldd_a):
    """Test that CLDD-A only contains the documented boundaries, strategies, and detectors[cite: 1]."""
    assert set(cldd_a["dataset"].unique()) == set(DATASETS_A)
    assert set(cldd_a["boundary"].unique()) == set(BOUNDARIES)
    assert set(cldd_a["strategy"].unique()) == set(STRATEGIES_A)
    assert set(cldd_a["detector"].unique()) == set(DETECTORS_A)


# --- TESTS FOR CLDD-B ---


def test_cldd_b_columns(cldd_b):
    """Test that CLDD-B contains all documented columns[cite: 1]."""
    assert list(cldd_b.columns) == EXPECTED_COLUMNS, (
        "CLDD-B columns do not match the datasheet."
    )


def test_cldd_b_factorial_size(cldd_b):
    """
    CLDD-B is a full factorial of 3 boundaries × 5 strategies × 5 detectors × 10 seeds.
    Total count should be 750[cite: 1].
    Note: The datasheet text states 'Only 5 seeds exists' but the factorial equation
    dictates '10 seeds = 750 configurations'[cite: 1]. This test asserts the mathematical total.
    """
    assert len(cldd_b) == 750, "CLDD-B should have exactly 750 configurations."


def test_cldd_b_unique_values(cldd_b):
    """Test that CLDD-B only contains the documented boundaries, strategies, and detectors[cite: 1]."""
    assert set(cldd_b["boundary"].unique()) == set(BOUNDARIES)
    assert set(cldd_b["strategy"].unique()) == set(STRATEGIES_B)
    assert set(cldd_b["detector"].unique()) == set(DETECTORS_B)


# --- SHARED DATA TYPE & LENGTH TESTS ---


def test_cldd_data_types(cldd_a):
    """Validate data types based on the column definitions in the datasheet[cite: 1]."""
    assert cldd_a["seed"].dtype == np.int32
    assert cldd_a["max_delay"].dtype == np.int32
    assert cldd_a["max_early"].dtype == np.int32

    # Verify stream columns exist as objects (lists in pandas)[cite: 1]
    assert isinstance(cldd_a["error_stream"].iloc[0], (list, np.ndarray))
    assert isinstance(cldd_a["trues"].iloc[0], (list, np.ndarray))
    assert isinstance(cldd_a["preds"].iloc[0], (list, np.ndarray))


def test_stream_lengths(cldd_a):
    """
    Check if the error_stream list has roughly the stream length of its dataset[cite: 1].
    We allow a small tolerance window for the word 'Roughly'[cite: 1].
    """
    for dataset, length in STREAM_LENGTHS.items():
        stream = cldd_a.loc[cldd_a["dataset"] == dataset, "error_stream"].iloc[0]
        assert 0.97 * length <= len(stream) <= 1.03 * length, (
            f"{dataset} error_stream length is not roughly {length}."
        )


def test_no_duplicate_seeds(cldd_a, cldd_b):
    """Test that every combination of boundary, strategy, detector, and seed is unique[cite: 1]."""
    assert not cldd_a.duplicated(
        subset=["dataset", "boundary", "strategy", "detector", "seed"]
    ).any(), "CLDD-A contains duplicate configurations."
    assert not cldd_b.duplicated(
        subset=["dataset", "boundary", "strategy", "detector", "seed"]
    ).any(), "CLDD-B contains duplicate configurations."
