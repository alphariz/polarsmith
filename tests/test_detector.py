import polars as pl
import pytest
import warnings
from polarsmith._detector import detect_smart_flags


@pytest.fixture
def df_mixed():
    """DataFrame dengan berbagai tipe kolom."""
    from datetime import datetime
    return pl.DataFrame({
        "age":       [25, 30, 45],
        "income":    [50000.0, 80000.0, 120000.0],
        "category":  ["A", "B", "A"],
        "timestamp": pl.Series([
            datetime(2023, 1, 1),
            datetime(2023, 6, 1),
            datetime(2023, 12, 1),
        ]).cast(pl.Datetime),
        "churn":     [0, 1, 0],
    })


def test_smart_detects_auto_bin_for_numeric(df_mixed):
    flags = detect_smart_flags(df_mixed, "churn")
    assert flags["auto_bin"] is True


def test_smart_detects_cyclical_for_datetime(df_mixed):
    flags = detect_smart_flags(df_mixed, "churn")
    assert flags["cyclical"] is True


def test_smart_detects_interactions_for_multi_numeric(df_mixed):
    flags = detect_smart_flags(df_mixed, "churn")
    assert flags["interactions"] is True


def test_smart_detects_target_encoding_with_target(df_mixed):
    flags = detect_smart_flags(df_mixed, "churn")
    assert flags["target_encoding"] is True


def test_smart_no_target_encoding_without_target(df_mixed):
    flags = detect_smart_flags(df_mixed, None)
    assert flags["target_encoding"] is False


def test_smart_warns_about_categoricals_without_target(df_mixed):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detect_smart_flags(df_mixed, None)
        messages = [str(x.message).lower() for x in w]
        assert any("target" in m for m in messages)


def test_smart_warns_high_null_column():
    df = pl.DataFrame({
        "a": [1.0, None, None, None, None],
        "b": [1, 2, 3, 4, 5],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detect_smart_flags(df, None)
        messages = [str(x.message) for x in w]
        assert any("missing" in m.lower() or "null" in m.lower() for m in messages)


def test_smart_only_numeric_no_cyclical(df_numeric):
    flags = detect_smart_flags(df_numeric, None)
    assert flags["cyclical"] is False


def test_smart_single_numeric_no_interactions():
    df = pl.DataFrame({"a": [1, 2, 3]})
    flags = detect_smart_flags(df, None)
    assert flags["interactions"] is False