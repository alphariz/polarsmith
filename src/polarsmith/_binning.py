"""
_binning.py — Auto binning untuk kolom numerik.

Mendukung tiga metode:
- equal_width  : bin dengan lebar seragam (pl.cut)
- equal_freq   : bin dengan frekuensi seragam (pl.qcut)
- quantile     : alias equal_freq, lebih eksplisit
"""
from __future__ import annotations

import polars as pl


_DEFAULT_MAX_BINS = 10
_DEFAULT_METHOD = "equal_width"
_SUPPORTED_METHODS = {"equal_width", "equal_freq", "quantile"}


def bin_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Tambahkan kolom binned untuk setiap kolom numerik.

    Kolom baru dinamai: `{original_col}_bin`
    Tipe output: String (label kategori bin)

    Parameters
    ----------
    df : pl.DataFrame
    config : dict
        Kunci yang didukung:
        - "max_bins" (int, default 10)
        - "method" (str, default "equal_width")
        - "columns" (list[str], optional) — jika None, semua kolom numerik

    Returns
    -------
    pl.DataFrame dengan kolom `{col}_bin` tambahan.
    """
    max_bins: int = config.get("max_bins", _DEFAULT_MAX_BINS)
    method: str = config.get("method", _DEFAULT_METHOD)
    target_cols: list[str] | None = config.get("columns", None)

    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Binning method harus salah satu dari {_SUPPORTED_METHODS}, "
            f"dapat: '{method}'"
        )

    numeric_cols = _get_numeric_cols(df, target_cols)
    if not numeric_cols:
        return df

    new_cols: list[pl.Series] = []
    for col_name in numeric_cols:
        series = df[col_name]
        binned = _bin_series(series, max_bins, method)
        new_cols.append(binned)

    return df.with_columns(new_cols)


def _get_numeric_cols(df: pl.DataFrame, target_cols: list[str] | None) -> list[str]:
    """Kembalikan daftar kolom numerik yang akan di-bin."""
    numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.Float32, pl.Float64)
    all_numeric = [
        col for col in df.columns
        if df[col].dtype in numeric_dtypes
    ]
    if target_cols is not None:
        invalid = set(target_cols) - set(df.columns)
        if invalid:
            raise ValueError(f"Kolom tidak ditemukan di DataFrame: {invalid}")
        return [c for c in target_cols if c in all_numeric]
    return all_numeric


def _bin_series(series: pl.Series, max_bins: int, method: str) -> pl.Series:
    """
    Bin satu Series, return Series baru dengan nama `{original}_bin`.

    Menangani edge case:
    - Semua nilai sama → return semua label "same_value"
    - Nilai unik < max_bins → kurangi jumlah bin otomatis
    """
    output_name = f"{series.name}_bin"
    n_unique = series.n_unique()

    if n_unique <= 1:
        return pl.Series(output_name, ["same_value"] * len(series))

    actual_bins = min(max_bins, n_unique)

    try:
        if method == "equal_width":
            min_v = series.min()
            max_v = series.max()

            # Handle case where all values are same or null
            if min_v == max_v or min_v is None or max_v is None:
                return pl.Series(output_name, ["same_value"] * len(series))

            # Calculate breaks for equal width
            # actual_bins bins need actual_bins - 1 interior breaks
            step = (max_v - min_v) / actual_bins
            breaks = [min_v + i * step for i in range(1, actual_bins)]

            return (
                series.cut(breaks)
                      .cast(pl.String)
                      .alias(output_name)
            )
        else:  # equal_freq atau quantile
            return (
                series.qcut(actual_bins, allow_duplicates=True)
                      .cast(pl.String)
                      .alias(output_name)
            )

    except Exception as e:
        # Fallback: jika cut gagal (misal semua NaN), return null series
        import warnings
        warnings.warn(
            f"Binning gagal untuk kolom '{series.name}': {e}. Kolom di-skip.",
            UserWarning,
            stacklevel=4,
        )
        return pl.Series(output_name, [None] * len(series), dtype=pl.String)