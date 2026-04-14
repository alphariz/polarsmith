"""
_encoding.py — Target encoding untuk kolom kategorik.

Mendukung dua metode:
1. WoE (Weight of Evidence): log(P(X|Y=1) / P(X|Y=0))
   - Cocok untuk binary classification
   - Output: float, interpretable
   
2. James-Stein: shrinkage estimator
   - Formula: λ × mean_group + (1 - λ) × mean_global
   - λ (shrinkage) makin besar jika group makin kecil → pull ke global mean
   - Robust terhadap rare categories

Keduanya diimplementasikan dengan fold-safe strategy:
- Encoding dihitung di luar fold yang sedang diencode
- Mencegah target leakage saat training
"""
from __future__ import annotations

import warnings
import polars as pl


_DEFAULT_METHOD = "james_stein"
_DEFAULT_N_FOLDS = 5
_SMOOTHING_PRIOR = 1.0   # untuk WoE smoothing (pseudo-count)


def add_target_encoding(
    df: pl.DataFrame,
    target: str,
    config: dict,
) -> pl.DataFrame:
    """
    Tambahkan target-encoded columns untuk semua kolom kategorik.

    Kolom baru dinamai: `{col}_enc_{method}`

    Parameters
    ----------
    df : pl.DataFrame
    target : str
        Nama kolom target (harus numerik: 0/1 untuk WoE, kontinu untuk JS).
    config : dict
        - "method": "james_stein" (default) atau "woe"
        - "n_folds": int (default 5) — jumlah fold untuk leakage prevention
        - "columns": list[str] — kolom kategorik yang di-encode (None = semua)

    Returns
    -------
    pl.DataFrame dengan kolom encoded tambahan.
    """
    method: str = config.get("method", _DEFAULT_METHOD)
    n_folds: int = config.get("n_folds", _DEFAULT_N_FOLDS)
    target_cols_cfg: list[str] | None = config.get("columns", None)

    if method not in ("james_stein", "woe"):
        raise ValueError(
            f"Encoding method harus 'james_stein' atau 'woe', dapat: '{method}'"
        )

    if target not in df.columns:
        raise ValueError(f"Kolom target '{target}' tidak ditemukan di DataFrame.")

    cat_cols = _get_categorical_cols(df, target, target_cols_cfg)
    if not cat_cols:
        warnings.warn(
            "Tidak ada kolom kategorik ditemukan untuk target encoding.",
            UserWarning,
            stacklevel=3,
        )
        return df

    suffix = f"_enc_{method}"
    encode_fn = _james_stein_encode if method == "james_stein" else _woe_encode

    new_cols: list[pl.Series] = []
    for col in cat_cols:
        encoded = encode_fn(df[col], df[target], n_folds)
        new_cols.append(encoded.alias(f"{col}{suffix}"))

    return df.with_columns(new_cols)


# ---------- James-Stein ----------

def _james_stein_encode(
    feature: pl.Series,
    target: pl.Series,
    n_folds: int,
) -> pl.Series:
    """
    James-Stein estimator dengan fold-safe cross-encoding.

    Untuk setiap fold k:
      - Hitung mean per group di fold LAIN (out-of-fold)
      - Apply shrinkage: λ_g × mean_g + (1 - λ_g) × mean_global
      - λ_g = n_g / (n_g + C) dimana C adalah smoothing constant

    Smoothing constant C = variance_global / variance_between_groups
    """
    output_name = feature.name
    n = len(feature)
    global_mean = target.mean()

    # Buat fold assignments
    fold_idx = _assign_folds(n, n_folds)
    encoded = pl.Series([0.0] * n)

    # Hitung smoothing constant C dari keseluruhan data
    # C besar → shrink lebih kuat ke global mean
    smoothing_c = _compute_smoothing_constant(feature, target, global_mean)

    for fold in range(n_folds):
        # Mask: baris yang ada di fold ini (akan di-encode)
        in_fold = (fold_idx == fold)
        out_fold = ~in_fold

        # Hitung statistik dari out-of-fold data
        feat_out = feature.filter(out_fold)
        tgt_out = target.filter(out_fold)

        group_stats = _compute_group_stats_js(feat_out, tgt_out, global_mean, smoothing_c)

        # Apply encoding ke baris yang ada di fold ini
        feat_in_fold = feature.filter(in_fold)
        fold_encoded = feat_in_fold.map_elements(
            lambda v: group_stats.get(v, global_mean),
            return_dtype=pl.Float64,
        )

        # Insert ke posisi yang benar
        in_fold_indices = [i for i, flag in enumerate(in_fold.to_list()) if flag]
        encoded_list = encoded.to_list()
        for idx, val in zip(in_fold_indices, fold_encoded.to_list()):
            encoded_list[idx] = val if val is not None else global_mean
        encoded = pl.Series(encoded_list)

    return encoded.alias(output_name)


def _compute_smoothing_constant(
    feature: pl.Series,
    target: pl.Series,
    global_mean: float,
) -> float:
    """
    Hitung C = variance_target / variance_between_groups.
    Jika variance_between kecil → C besar → shrink kuat (rare category).
    """
    df_temp = pl.DataFrame({"f": feature, "t": target})
    group_means = (
        df_temp.group_by("f")
        .agg(pl.col("t").mean().alias("mean_g"), pl.col("t").count().alias("n_g"))
    )
    var_global = float(target.var() or 1.0)
    var_between = float(group_means["mean_g"].var() or 1e-6)
    return var_global / (var_between + 1e-10)


def _compute_group_stats_js(
    feature: pl.Series,
    target: pl.Series,
    global_mean: float,
    smoothing_c: float,
) -> dict[str, float]:
    """
    Hitung James-Stein estimate per group dari out-of-fold data.
    Return dict {category_value: encoded_value}.
    """
    df_temp = pl.DataFrame({"f": feature, "t": target})
    stats = (
        df_temp.group_by("f")
        .agg([
            pl.col("t").mean().alias("mean_g"),
            pl.col("t").count().alias("n_g"),
        ])
    )
    result = {}
    for row in stats.iter_rows(named=True):
        n_g = row["n_g"]
        mean_g = row["mean_g"]
        # λ = n_g / (n_g + C): makin kecil group → λ makin kecil → lebih shrink
        lam = n_g / (n_g + smoothing_c)
        js_estimate = lam * mean_g + (1 - lam) * global_mean
        result[row["f"]] = js_estimate
    return result


# ---------- Weight of Evidence ----------

def _woe_encode(
    feature: pl.Series,
    target: pl.Series,
    n_folds: int,
) -> pl.Series:
    """
    WoE encoding dengan fold-safe cross-encoding.

    WoE_g = log( P(X=g | Y=1) / P(X=g | Y=0) )
           = log( (n_g1 / n_total1) / (n_g0 / n_total0) )

    Dengan Laplace smoothing untuk menghindari log(0).
    """
    output_name = feature.name
    n = len(feature)
    fold_idx = _assign_folds(n, n_folds)
    encoded = pl.Series([0.0] * n)

    for fold in range(n_folds):
        in_fold = (fold_idx == fold)
        out_fold = ~in_fold

        feat_out = feature.filter(out_fold)
        tgt_out = target.filter(out_fold)

        woe_map = _compute_woe_map(feat_out, tgt_out)

        feat_in_fold = feature.filter(in_fold)
        fold_encoded = feat_in_fold.map_elements(
            lambda v: woe_map.get(v, 0.0),
            return_dtype=pl.Float64,
        )

        in_fold_indices = [i for i, flag in enumerate(in_fold.to_list()) if flag]
        encoded_list = encoded.to_list()
        for idx, val in zip(in_fold_indices, fold_encoded.to_list()):
            encoded_list[idx] = val if val is not None else 0.0
        encoded = pl.Series(encoded_list)

    return encoded.alias(output_name)


def _compute_woe_map(feature: pl.Series, target: pl.Series) -> dict[str, float]:
    """
    Hitung WoE per category dari data.
    Menggunakan Laplace smoothing (+ prior) untuk menghindari log(0).
    """
    import math
    df_temp = pl.DataFrame({"f": feature, "t": target})

    n_total_1 = float(target.sum() or 1)
    n_total_0 = float(len(target) - n_total_1 or 1)

    stats = (
        df_temp.group_by("f")
        .agg([
            pl.col("t").sum().alias("n_1"),
            pl.col("t").count().alias("n_total"),
        ])
        .with_columns(
            (pl.col("n_total") - pl.col("n_1")).alias("n_0")
        )
    )

    woe_map = {}
    for row in stats.iter_rows(named=True):
        # Laplace smoothing
        p_g1 = (row["n_1"] + _SMOOTHING_PRIOR) / (n_total_1 + _SMOOTHING_PRIOR)
        p_g0 = (row["n_0"] + _SMOOTHING_PRIOR) / (n_total_0 + _SMOOTHING_PRIOR)
        woe_map[row["f"]] = math.log(p_g1 / p_g0)
    return woe_map


# ---------- Helpers ----------

def _assign_folds(n: int, n_folds: int) -> pl.Series:
    """Assign fold index secara sekuensial (bukan random) untuk reproducibility."""
    return pl.Series("fold", [i % n_folds for i in range(n)])


def _get_categorical_cols(
    df: pl.DataFrame,
    target: str,
    target_cols_cfg: list[str] | None,
) -> list[str]:
    """Kembalikan kolom kategorik/string, kecuali kolom target."""
    cat_dtypes = (pl.String, pl.Categorical, pl.Utf8)
    all_cat = [
        c for c in df.columns
        if df[c].dtype in cat_dtypes and c != target
    ]
    if target_cols_cfg is not None:
        return [c for c in target_cols_cfg if c in all_cat]
    return all_cat