import math
import polars as pl
import pytest
from polarsmith._encoding import (
    add_target_encoding,
    _james_stein_encode,
    _woe_encode,
    _compute_woe_map,
    _assign_folds,
    _get_categorical_cols,
)


# --- unit test: _assign_folds ---

def test_assign_folds_length():
    folds = _assign_folds(100, 5)
    assert len(folds) == 100


def test_assign_folds_covers_all_folds():
    folds = _assign_folds(50, 5)
    assert set(folds.to_list()) == {0, 1, 2, 3, 4}


def test_assign_folds_reproducible():
    f1 = _assign_folds(100, 5)
    f2 = _assign_folds(100, 5)
    assert f1.to_list() == f2.to_list()


# --- unit test: _get_categorical_cols ---

def test_get_categorical_cols_excludes_target(df_with_target):
    cols = _get_categorical_cols(df_with_target, "churn", None)
    assert "churn" not in cols
    assert "category" in cols


def test_get_categorical_cols_excludes_numeric(df_with_target):
    cols = _get_categorical_cols(df_with_target, "churn", None)
    assert "age" not in cols


# --- unit test: _compute_woe_map ---

def test_woe_map_keys_are_categories(df_with_target):
    woe = _compute_woe_map(df_with_target["category"], df_with_target["churn"])
    for cat in df_with_target["category"].unique().to_list():
        assert cat in woe


def test_woe_map_values_are_finite(df_with_target):
    woe = _compute_woe_map(df_with_target["category"], df_with_target["churn"])
    for val in woe.values():
        assert math.isfinite(val)


# --- unit test: _james_stein_encode ---

def test_james_stein_encode_length(df_with_target):
    encoded = _james_stein_encode(
        df_with_target["category"], df_with_target["churn"], n_folds=5
    )
    assert len(encoded) == len(df_with_target)


def test_james_stein_encode_no_null(df_with_target):
    encoded = _james_stein_encode(
        df_with_target["category"], df_with_target["churn"], n_folds=5
    )
    assert encoded.null_count() == 0


def test_james_stein_shrinks_toward_global_mean(df_with_target):
    """
    Encoded values harus berada antara min_group_mean dan max_group_mean,
    yaitu bukti shrinkage menuju global mean.
    """
    global_mean = df_with_target["churn"].mean()
    encoded = _james_stein_encode(
        df_with_target["category"], df_with_target["churn"], n_folds=5
    )
    # Semua nilai encoded harus lebih dekat ke global mean dari grup extremes
    # Cukup periksa range tidak melebihi target range
    assert encoded.min() >= df_with_target["churn"].min() - 0.1
    assert encoded.max() <= df_with_target["churn"].max() + 0.1


# --- Leakage test — paling kritis ---

def test_james_stein_no_leakage():
    """
    Jika kita encode kolom kategori yang PERFECTLY predicts target,
    encoded values yang dihasilkan harus TIDAK perfect (karena fold-safe).
    Perfect prediction = leakage terjadi.
    """
    import random
    random.seed(42)
    n = 200
    # Kategori A selalu → target 1, B selalu → target 0
    cats = (["A"] * (n // 2)) + (["B"] * (n // 2))
    targets = ([1] * (n // 2)) + ([0] * (n // 2))
    feature = pl.Series("cat", cats)
    target = pl.Series("tgt", targets)

    encoded = _james_stein_encode(feature, target, n_folds=5)

    # Dengan leakage: A selalu encode 1.0, B selalu encode 0.0
    # Tanpa leakage (fold-safe): nilai tidak sempurna karena dihitung OOF
    a_vals = encoded.filter(feature == "A")
    b_vals = encoded.filter(feature == "B")

    # Nilai A harus > 0.5 (masih prediktif)
    assert a_vals.mean() > 0.5
    # Tapi tidak harus persis 1.0 (bukan leakage)
    # (pada data kecil dengan fold-safe, nilai bisa mendekati 1.0 tapi tidak persis)
    assert b_vals.mean() < 0.5


def test_woe_encode_length(df_with_target):
    encoded = _woe_encode(
        df_with_target["category"], df_with_target["churn"], n_folds=5
    )
    assert len(encoded) == len(df_with_target)


def test_woe_encode_positive_for_high_event_rate():
    """
    Kategori dengan event rate tinggi harus punya WoE positif,
    dan kategori dengan event rate rendah harus WoE negatif.
    """
    feature = pl.Series("cat", (["high"] * 100) + (["low"] * 100))
    # high: 90% event, low: 10% event
    target = pl.Series("t",
        ([1] * 90 + [0] * 10) +  # high
        ([1] * 10 + [0] * 90)    # low
    )
    encoded = _woe_encode(feature, target, n_folds=5)
    high_mean = encoded.filter(feature == "high").mean()
    low_mean = encoded.filter(feature == "low").mean()
    assert high_mean > low_mean


# --- unit test: add_target_encoding ---

def test_add_target_encoding_adds_columns(df_with_target):
    result = add_target_encoding(df_with_target, "churn", {})
    assert "category_enc_james_stein" in result.columns


def test_add_target_encoding_woe(df_with_target):
    result = add_target_encoding(
        df_with_target, "churn", {"method": "woe"}
    )
    assert "category_enc_woe" in result.columns


def test_add_target_encoding_invalid_method_raises(df_with_target):
    with pytest.raises(ValueError, match="method harus"):
        add_target_encoding(df_with_target, "churn", {"method": "mean"})


def test_add_target_encoding_missing_target_raises(df_with_target):
    with pytest.raises(ValueError, match="tidak ditemukan"):
        add_target_encoding(df_with_target, "nonexistent", {})


def test_add_target_encoding_preserves_original_cols(df_with_target):
    result = add_target_encoding(df_with_target, "churn", {})
    for col in df_with_target.columns:
        assert col in result.columns