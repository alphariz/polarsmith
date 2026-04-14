"""
_cyclical.py — Cyclical encoding untuk fitur berbasis waktu.

Mengubah nilai periodik menjadi pasangan sin/cos sehingga jarak
antara nilai "23" dan "0" (jam) tetap kecil, bukan 23 unit.

Formula: sin(2π × value / period), cos(2π × value / period)
"""
from __future__ import annotations

import math
import polars as pl


# Mapping nama komponen waktu → period-nya
_DATETIME_COMPONENTS: dict[str, int] = {
    "hour":       24,
    "minute":     60,
    "second":     60,
    "dayofweek":  7,
    "dayofmonth": 31,
    "dayofyear":  365,
    "month":      12,
    "weekofyear": 52,
}

_DEFAULT_COMPONENTS = ["hour", "dayofweek", "month"]


def add_cyclical_features(
    df: pl.DataFrame,
    config: list[str] | None,
) -> pl.DataFrame:
    """
    Tambahkan sin/cos encoding untuk setiap komponen waktu yang diminta.

    Cara kerja:
    1. Cari kolom bertipe Datetime di DataFrame.
    2. Ekstrak komponen waktu (hour, dayofweek, dll).
    3. Hitung sin dan cos untuk setiap komponen.
    4. Tambahkan sebagai kolom baru: `{col}_{component}_sin` dan `_cos`.

    Parameters
    ----------
    df : pl.DataFrame
    config : list[str] or None
        Daftar komponen yang diinginkan, misal ["hour", "dayofweek"].
        Jika None, pakai default: ["hour", "dayofweek", "month"].

    Returns
    -------
    pl.DataFrame dengan kolom sin/cos tambahan.
    """
    components = _resolve_components(config)
    datetime_cols = _get_datetime_cols(df)

    if not datetime_cols:
        return df

    new_expressions: list[pl.Expr] = []
    for dt_col in datetime_cols:
        for component in components:
            if component not in _DATETIME_COMPONENTS:
                continue
            period = _DATETIME_COMPONENTS[component]
            extracted = _extract_component(dt_col, component)
            sin_col = (
                (extracted * (2 * math.pi / period)).sin()
                .alias(f"{dt_col}_{component}_sin")
            )
            cos_col = (
                (extracted * (2 * math.pi / period)).cos()
                .alias(f"{dt_col}_{component}_cos")
            )
            new_expressions.extend([sin_col, cos_col])

    if not new_expressions:
        return df

    return df.with_columns(new_expressions)


def _resolve_components(config: list[str] | None) -> list[str]:
    """Validasi dan normalisasi daftar komponen."""
    if config is None:
        return _DEFAULT_COMPONENTS
    invalid = set(config) - set(_DATETIME_COMPONENTS)
    if invalid:
        raise ValueError(
            f"Komponen tidak dikenal: {invalid}. "
            f"Pilihan valid: {set(_DATETIME_COMPONENTS)}"
        )
    return config


def _get_datetime_cols(df: pl.DataFrame) -> list[str]:
    """Kembalikan nama kolom dengan dtype Datetime atau Date."""
    return [
        col for col in df.columns
        if df[col].dtype in (pl.Datetime, pl.Date)
    ]


def _extract_component(col_name: str, component: str) -> pl.Expr:
    """
    Ekstrak komponen waktu dari kolom sebagai pl.Expr Float64.

    Polars menggunakan .dt accessor untuk semua operasi datetime.
    """
    col = pl.col(col_name)
    mapping = {
        "hour":       col.dt.hour(),
        "minute":     col.dt.minute(),
        "second":     col.dt.second(),
        "dayofweek":  col.dt.weekday(),
        "dayofmonth": col.dt.day(),
        "dayofyear":  col.dt.ordinal_day(),
        "month":      col.dt.month(),
        "weekofyear": col.dt.week(),
    }
    return mapping[component].cast(pl.Float64)