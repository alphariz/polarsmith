"""
_interactions.py — Pembuatan interaction terms antar kolom numerik.

Dua jenis interaksi:
- multiply : col_a * col_b  → menangkap efek multiplikatif
- ratio    : col_a / col_b  → menangkap efek relatif (dengan guard div/0)
"""
from __future__ import annotations

import itertools
import warnings
import polars as pl


_DEFAULT_MAX_PAIRS = 10
_DEFAULT_DEGREE = 2   # hanya pairwise, bukan triple dst


def add_interactions(
    df: pl.DataFrame,
    config: list[str] | dict | None,
) -> pl.DataFrame:
    """
    Tambahkan kolom interaction terms antar kolom numerik.

    Kolom baru:
    - multiply: `{a}_x_{b}`
    - ratio:    `{a}_div_{b}` (skip jika b semua nol)

    Parameters
    ----------
    df : pl.DataFrame
    config : list[str], dict, or None
        - None: auto-detect semua pasangan numerik (max `max_pairs` pasang)
        - list[str]: daftar interaksi eksplisit, format "col_a*col_b"
          misal: ["age*income", "tenure*score"]
        - dict: {"pairs": [...], "max_pairs": int, "add_ratio": bool}

    Returns
    -------
    pl.DataFrame dengan kolom interaksi tambahan.
    """
    cfg = _parse_config(config)
    numeric_cols = _get_numeric_cols(df)

    if len(numeric_cols) < 2:
        return df

    pairs = _resolve_pairs(cfg, numeric_cols, df)
    if not pairs:
        return df

    new_exprs: list[pl.Expr] = []
    for col_a, col_b in pairs:
        # Multiply
        new_exprs.append(
            (pl.col(col_a) * pl.col(col_b)).alias(f"{col_a}_x_{col_b}")
        )
        # Ratio (opsional, default True)
        if cfg.get("add_ratio", True):
            b_series = df[col_b]
            if b_series.abs().max() > 1e-10:
                new_exprs.append(
                    (pl.col(col_a) / pl.col(col_b).replace(0, None))
                    .alias(f"{col_a}_div_{col_b}")
                )
            else:
                warnings.warn(
                    f"Ratio {col_a}/{col_b} di-skip karena {col_b} mendekati nol.",
                    UserWarning,
                    stacklevel=3,
                )

    return df.with_columns(new_exprs)


def _parse_config(config: list[str] | dict | None) -> dict:
    """Normalisasi semua bentuk config ke dict."""
    if config is None:
        return {"max_pairs": _DEFAULT_MAX_PAIRS, "add_ratio": True}
    if isinstance(config, list):
        return {"pairs": config, "max_pairs": len(config), "add_ratio": True}
    return {"max_pairs": _DEFAULT_MAX_PAIRS, "add_ratio": True, **config}


def _get_numeric_cols(df: pl.DataFrame) -> list[str]:
    numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.Float32, pl.Float64)
    return [c for c in df.columns if df[c].dtype in numeric_dtypes]


def _resolve_pairs(
    cfg: dict,
    numeric_cols: list[str],
    df: pl.DataFrame,
) -> list[tuple[str, str]]:
    """
    Tentukan pasangan kolom yang akan diinteraksikan.
    Jika config eksplisit ada, parse dan validasi.
    Jika tidak, auto-generate hingga max_pairs.
    """
    if "pairs" in cfg:
        return _parse_explicit_pairs(cfg["pairs"], df.columns)

    max_pairs: int = cfg.get("max_pairs", _DEFAULT_MAX_PAIRS)
    all_pairs = list(itertools.combinations(numeric_cols, 2))
    return all_pairs[:max_pairs]


def _parse_explicit_pairs(
    pairs: list[str],
    available_cols: list[str],
) -> list[tuple[str, str]]:
    """
    Parse string "col_a*col_b" menjadi tuple (col_a, col_b).
    Validasi bahwa kedua kolom ada di DataFrame.
    """
    result = []
    for pair_str in pairs:
        if "*" not in pair_str:
            raise ValueError(
                f"Format pair tidak valid: '{pair_str}'. "
                f"Gunakan format 'col_a*col_b'."
            )
        col_a, col_b = [s.strip() for s in pair_str.split("*", 1)]
        for col in (col_a, col_b):
            if col not in available_cols:
                raise ValueError(
                    f"Kolom '{col}' tidak ditemukan di DataFrame. "
                    f"Kolom tersedia: {available_cols}"
                )
        result.append((col_a, col_b))
    return result