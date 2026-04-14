import polars as pl
import pytest
from polarsmith._interactions import (
    add_interactions,
    _parse_config,
    _parse_explicit_pairs,
    _resolve_pairs,
    _get_numeric_cols,
)


# --- unit test: _parse_config ---

def test_parse_config_none_returns_defaults():
    cfg = _parse_config(None)
    assert cfg["max_pairs"] == 10
    assert cfg["add_ratio"] is True


def test_parse_config_list_wraps_as_pairs():
    cfg = _parse_config(["age*income"])
    assert cfg["pairs"] == ["age*income"]


def test_parse_config_dict_passthrough():
    cfg = _parse_config({"max_pairs": 5, "add_ratio": False})
    assert cfg["max_pairs"] == 5
    assert cfg["add_ratio"] is False


# --- unit test: _parse_explicit_pairs ---

def test_parse_explicit_pairs_valid():
    pairs = _parse_explicit_pairs(["age*income"], ["age", "income", "score"])
    assert pairs == [("age", "income")]


def test_parse_explicit_pairs_invalid_format_raises():
    with pytest.raises(ValueError, match="Format pair"):
        _parse_explicit_pairs(["age-income"], ["age", "income"])


def test_parse_explicit_pairs_missing_col_raises():
    with pytest.raises(ValueError, match="tidak ditemukan"):
        _parse_explicit_pairs(["age*nonexistent"], ["age", "income"])


# --- unit test: add_interactions ---

def test_interactions_adds_multiply_columns(df_numeric):
    result = add_interactions(df_numeric, ["age*income"])
    assert "age_x_income" in result.columns


def test_interactions_adds_ratio_columns(df_numeric):
    result = add_interactions(df_numeric, ["age*income"])
    assert "age_div_income" in result.columns


def test_interactions_multiply_values_correct(df_numeric):
    result = add_interactions(df_numeric, ["age*income"])
    expected = df_numeric["age"] * df_numeric["income"]
    assert (result["age_x_income"] - expected).abs().max() < 1e-6


def test_interactions_preserves_original_columns(df_numeric):
    result = add_interactions(df_numeric, ["age*income"])
    for col in df_numeric.columns:
        assert col in result.columns


def test_interactions_max_pairs_limits_output(df_numeric):
    result = add_interactions(df_numeric, {"max_pairs": 2, "add_ratio": False})
    new_cols = [c for c in result.columns if c not in df_numeric.columns]
    # max 2 pairs × 1 (no ratio) = 2 kolom baru
    assert len(new_cols) <= 2


def test_interactions_auto_detect_pairs(df_numeric):
    result = add_interactions(df_numeric, None)
    new_cols = [c for c in result.columns if "_x_" in c]
    assert len(new_cols) > 0


def test_interactions_too_few_numeric_cols_returns_unchanged():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = add_interactions(df, None)
    assert result.equals(df)


def test_interactions_no_ratio_flag(df_numeric):
    result = add_interactions(df_numeric, {"pairs": ["age*income"], "add_ratio": False})
    assert "age_div_income" not in result.columns
    assert "age_x_income" in result.columns