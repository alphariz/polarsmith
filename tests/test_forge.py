import polars as pl
import pytest
import warnings
from polarsmith import forge


def test_forge_returns_polars_dataframe(df_numeric):
    result = forge(df_numeric)
    assert isinstance(result, pl.DataFrame)


def test_forge_preserves_original_columns(df_numeric):
    original_cols = set(df_numeric.columns)
    result = forge(df_numeric)
    assert original_cols.issubset(set(result.columns))


def test_forge_strategy_full_activates_all(df_with_datetime):
    """strategy='full' harus menambah kolom dari cyclical minimal."""
    result = forge(df_with_datetime, strategy="full")
    assert result.shape[1] > df_with_datetime.shape[1]


def test_forge_strategy_minimal_no_extra_columns(df_numeric):
    result = forge(df_numeric, strategy="minimal")
    assert result.shape[1] == df_numeric.shape[1]


def test_forge_invalid_strategy_raises(df_numeric):
    with pytest.raises(ValueError, match="strategy harus"):
        forge(df_numeric, strategy="unknown")


def test_forge_target_encoding_without_target_warns(df_with_target):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forge(df_with_target, target_encoding=True)
        assert len(w) == 1
        assert "target" in str(w[0].message).lower()


def test_forge_accepts_pandas_dataframe(df_numeric):
    """Pastikan pandas DataFrame dikonversi tanpa error."""
    pandas_df = df_numeric.to_pandas()
    result = forge(pandas_df)
    assert isinstance(result, pl.DataFrame)


def test_forge_invalid_input_type_raises():
    with pytest.raises(TypeError):
        forge({"a": [1, 2, 3]})