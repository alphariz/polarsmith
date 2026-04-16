import math
import polars as pl
import pytest
from polarsmith._cyclical import (
    add_cyclical_features,
    _get_datetime_cols,
    _resolve_components,
    _extract_component,
)


# --- unit test: _resolve_components ---

def test_resolve_components_none_returns_default():
    result = _resolve_components(None)
    assert "hour" in result
    assert "dayofweek" in result


def test_resolve_components_custom():
    result = _resolve_components(["hour", "month"])
    assert result == ["hour", "month"]


def test_resolve_components_invalid_raises():
    with pytest.raises(ValueError, match="tidak dikenal"):
        _resolve_components(["hour", "invalid_component"])


# --- unit test: _get_datetime_cols ---

def test_get_datetime_cols_detects_datetime(df_with_datetime):
    cols = _get_datetime_cols(df_with_datetime)
    assert "timestamp" in cols


def test_get_datetime_cols_ignores_numeric(df_numeric):
    cols = _get_datetime_cols(df_numeric)
    assert cols == []


# --- unit test: add_cyclical_features ---

def test_cyclical_adds_sin_cos_columns(df_with_datetime):
    result = add_cyclical_features(df_with_datetime, ["hour"])
    assert "timestamp_hour_sin" in result.columns
    assert "timestamp_hour_cos" in result.columns


def test_cyclical_sin_cos_range(df_with_datetime):
    """Nilai sin dan cos harus selalu dalam [-1, 1]."""
    result = add_cyclical_features(df_with_datetime, ["hour", "month"])
    for col in result.columns:
        if col.endswith("_sin") or col.endswith("_cos"):
            series = result[col].drop_nulls()
            assert series.min() >= -1.0 - 1e-9
            assert series.max() <= 1.0 + 1e-9


def test_cyclical_hour_midnight_and_23_close():
    """
    Jam 0 dan jam 23 seharusnya 'dekat' secara siklikal.
    Jarak euclidean (sin,cos) antara keduanya harus < jarak jam 0 dan jam 12.
    """
    df = pl.DataFrame({
        "ts": pl.Series([
            "2023-01-01 00:00:00",
            "2023-01-01 23:00:00",
            "2023-01-01 12:00:00",
        ]).str.to_datetime(),
    })
    result = add_cyclical_features(df, ["hour"])
    sin_vals = result["ts_hour_sin"].to_list()
    cos_vals = result["ts_hour_cos"].to_list()

    def dist(i, j):
        return math.sqrt((sin_vals[i]-sin_vals[j])**2 + (cos_vals[i]-cos_vals[j])**2)

    # Jarak jam 0 dan 23 harus lebih kecil dari jarak jam 0 dan 12
    assert dist(0, 1) < dist(0, 2)


def test_cyclical_no_datetime_cols_returns_unchanged(df_numeric):
    result = add_cyclical_features(df_numeric, None)
    assert result.equals(df_numeric)


def test_cyclical_preserves_original_columns(df_with_datetime):
    result = add_cyclical_features(df_with_datetime, ["hour"])
    for col in df_with_datetime.columns:
        assert col in result.columns


def test_cyclical_multiple_components(df_with_datetime):
    result = add_cyclical_features(df_with_datetime, ["hour", "dayofweek", "month"])
    for comp in ["hour", "dayofweek", "month"]:
        assert f"timestamp_{comp}_sin" in result.columns
        assert f"timestamp_{comp}_cos" in result.columns


def test_cyclical_handles_date_dtype():
    """Kolom Date (bukan Datetime) juga harus terdeteksi dan diproses."""
    df = pl.DataFrame({
        "date_col": pl.Series(["2023-01-15", "2023-06-20"]).str.to_date(),
    })
    result = add_cyclical_features(df, ["month"])
    assert "date_col_month_sin" in result.columns


def test_cyclical_unknown_component_in_list_skipped():
    """
    Komponen yang tidak ada di _DATETIME_COMPONENTS di-skip,
    bukan raise error (validasi ada di level config).
    Untuk hit line 64, kita mock _resolve_components untuk bypass validation.
    """
    from unittest.mock import patch
    df_dt = pl.DataFrame({
        "ts": pl.Series(["2023-01-01 10:00:00"]).str.to_datetime(),
    })
    
    # Mocking _resolve_components untuk return ["unknown"] yang biasanya dilarang
    with patch("polarsmith._cyclical._resolve_components", return_value=["unknown"]):
        # Line 64: if component not in _DATETIME_COMPONENTS: continue
        # Line 78: if not new_expressions: return df
        result = add_cyclical_features(df_dt, ["not-actually-used-because-mocked"])
        assert result.equals(df_dt)
        assert "ts_unknown_sin" not in result.columns