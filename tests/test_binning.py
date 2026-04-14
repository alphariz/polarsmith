import polars as pl
import pytest
from polarsmith._binning import bin_features, _bin_series, _get_numeric_cols


# --- unit test: _get_numeric_cols ---

def test_get_numeric_cols_returns_only_numeric(df_numeric):
    cols = _get_numeric_cols(df_numeric, None)
    assert set(cols) == {"age", "income", "tenure", "score"}


def test_get_numeric_cols_excludes_string():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cols = _get_numeric_cols(df, None)
    assert cols == ["a"]


def test_get_numeric_cols_with_target_cols_filter(df_numeric):
    cols = _get_numeric_cols(df_numeric, ["age", "income"])
    assert cols == ["age", "income"]


def test_get_numeric_cols_invalid_target_raises(df_numeric):
    with pytest.raises(ValueError, match="tidak ditemukan"):
        _get_numeric_cols(df_numeric, ["nonexistent"])


# --- unit test: _bin_series ---

def test_bin_series_equal_width_produces_correct_name():
    s = pl.Series("age", list(range(100)))
    result = _bin_series(s, max_bins=5, method="equal_width")
    assert result.name == "age_bin"


def test_bin_series_returns_string_dtype():
    s = pl.Series("score", [float(i) for i in range(100)])
    result = _bin_series(s, max_bins=5, method="equal_width")
    assert result.dtype == pl.String


def test_bin_series_equal_freq():
    s = pl.Series("income", [float(i) for i in range(200)])
    result = _bin_series(s, max_bins=4, method="equal_freq")
    assert result.name == "income_bin"
    assert result.dtype == pl.String


def test_bin_series_quantile_alias():
    s = pl.Series("val", [float(i) for i in range(100)])
    result = _bin_series(s, max_bins=5, method="quantile")
    assert result.dtype == pl.String


def test_bin_series_constant_values_returns_same_value_label():
    s = pl.Series("const", [5.0] * 50)
    result = _bin_series(s, max_bins=10, method="equal_width")
    assert (result == "same_value").all()


def test_bin_series_fewer_unique_than_bins():
    """Jika nilai unik < max_bins, harus pakai actual_bins = n_unique."""
    s = pl.Series("cat", [1.0, 2.0, 3.0] * 50)
    result = _bin_series(s, max_bins=10, method="equal_width")
    assert result.null_count() == 0


# --- unit test: bin_features ---

def test_bin_features_adds_bin_columns(df_numeric):
    result = bin_features(df_numeric, {})
    for col in ["age", "income", "tenure", "score"]:
        assert f"{col}_bin" in result.columns


def test_bin_features_preserves_original_columns(df_numeric):
    result = bin_features(df_numeric, {})
    for col in df_numeric.columns:
        assert col in result.columns


def test_bin_features_respects_max_bins(df_numeric):
    result = bin_features(df_numeric, {"max_bins": 3})
    unique_bins = result["age_bin"].n_unique()
    assert unique_bins <= 3


def test_bin_features_invalid_method_raises(df_numeric):
    with pytest.raises(ValueError, match="method harus"):
        bin_features(df_numeric, {"method": "invalid"})


def test_bin_features_empty_dataframe():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
    result = bin_features(df, {})
    assert "a_bin" in result.columns


def test_bin_features_no_numeric_cols_returns_unchanged():
    df = pl.DataFrame({"name": ["Alice", "Bob"], "city": ["Jakarta", "Bandung"]})
    result = bin_features(df, {})
    assert result.equals(df)