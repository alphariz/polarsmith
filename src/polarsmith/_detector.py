"""
_detector.py — Smart detection kolom berdasarkan tipe dan konten.

Digunakan oleh forge() dengan strategy="smart" untuk otomatis
mengaktifkan fitur yang relevan tanpa perlu config manual.
"""
from __future__ import annotations

import warnings
import polars as pl


def detect_smart_flags(
    df: pl.DataFrame,
    target: str | None,
) -> dict[str, bool]:
    """
    Analisis DataFrame dan kembalikan rekomendasi flag yang sebaiknya aktif.

    Returns
    -------
    dict dengan keys: "auto_bin", "cyclical", "interactions", "target_encoding"
    """
    flags: dict[str, bool] = {
        "auto_bin":        False,
        "cyclical":        False,
        "interactions":    False,
        "target_encoding": False,
    }

    has_numeric   = _has_numeric_cols(df)
    has_datetime  = _has_datetime_cols(df)
    has_categ     = _has_categorical_cols(df, target)
    n_numeric     = _count_numeric_cols(df)

    if has_numeric:
        flags["auto_bin"] = True

    if has_datetime:
        flags["cyclical"] = True

    if has_numeric and n_numeric >= 2:
        flags["interactions"] = True

    if has_categ and target is not None:
        flags["target_encoding"] = True

    _warn_about_skipped(df, target, flags)
    return flags


def _has_numeric_cols(df: pl.DataFrame) -> bool:
    numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.Float32, pl.Float64)
    return any(df[c].dtype in numeric_dtypes for c in df.columns)


def _count_numeric_cols(df: pl.DataFrame) -> int:
    numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.Float32, pl.Float64)
    return sum(1 for c in df.columns if df[c].dtype in numeric_dtypes)


def _has_datetime_cols(df: pl.DataFrame) -> bool:
    return any(df[c].dtype in (pl.Datetime, pl.Date) for c in df.columns)


def _has_categorical_cols(df: pl.DataFrame, target: str | None) -> bool:
    cat_dtypes = (pl.String, pl.Categorical, pl.Utf8)
    return any(
        df[c].dtype in cat_dtypes and c != target
        for c in df.columns
    )


def _warn_about_skipped(
    df: pl.DataFrame,
    target: str | None,
    flags: dict[str, bool],
) -> None:
    """Beri tahu user kolom apa yang di-skip dan kenapa."""
    skipped = []

    # Kolom dengan terlalu banyak missing values
    for col in df.columns:
        null_pct = df[col].null_count() / len(df)
        if null_pct > 0.5:
            skipped.append(f"'{col}' ({null_pct:.0%} null)")

    if skipped:
        warnings.warn(
            f"Kolom berikut memiliki >50% missing values dan mungkin "
            f"menghasilkan fitur yang tidak informatif: {', '.join(skipped)}",
            UserWarning,
            stacklevel=4,
        )

    # Tidak ada target tapi ada kolom kategorik
    if _has_categorical_cols(df, target) and target is None:
        warnings.warn(
            "DataFrame memiliki kolom kategorik tapi `target` tidak disediakan. "
            "Target encoding tidak aktif. Sediakan target= untuk mengaktifkannya.",
            UserWarning,
            stacklevel=4,
        )