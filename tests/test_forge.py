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


def test_forge_with_target_encoding(df_with_target):
    """Hit lines 105-106 di _forge.py by providing target."""
    result = forge(
        df_with_target, target="churn", target_encoding=True
    )
    assert "category_enc_james_stein" in result.columns


# --- pandas compatibility tests ---

def test_forge_pandas_input_returns_polars_by_default(df_numeric):
    """Default behavior: input pandas → output tetap Polars."""
    pd_df = df_numeric.to_pandas()
    result = forge(pd_df)
    assert isinstance(result, pl.DataFrame)


def test_forge_pandas_input_return_pandas_flag(df_numeric):
    """return_pandas=True → output pandas."""
    pd_df = df_numeric.to_pandas()
    import pandas as pd
    result = forge(pd_df, return_pandas=True)
    assert isinstance(result, pd.DataFrame)


def test_forge_polars_input_return_pandas_flag(df_numeric):
    """Polars input + return_pandas=True → output pandas."""
    import pandas as pd
    result = forge(df_numeric, return_pandas=True)
    assert isinstance(result, pd.DataFrame)


def test_forge_pandas_roundtrip_preserves_shape(df_numeric):
    """Pandas → forge → pandas shape harus konsisten."""
    pd_df = df_numeric.to_pandas()
    result = forge(pd_df, auto_bin=True, return_pandas=True)
    assert result.shape[0] == pd_df.shape[0]
    assert result.shape[1] >= pd_df.shape[1]


def test_forge_invalid_input_type_raises():
    with pytest.raises(TypeError):
        forge({"a": [1, 2, 3]})