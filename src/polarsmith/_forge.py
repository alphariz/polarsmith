"""
_forge.py — Entry point utama Polarsmith.
Fungsi forge() menerima DataFrame dan mendispatch ke modul yang sesuai.
"""
from __future__ import annotations

import warnings
from typing import Any

import polars as pl


def forge(
    df: pl.DataFrame | Any,
    target: str | None = None,
    strategy: str = "smart",
    auto_bin: bool = False,
    cyclical: bool = False,
    interactions: bool = False,
    target_encoding: bool = False,
    return_pandas: bool = False,   
    config: dict | None = None,
) -> pl.DataFrame:
    """
    Automated feature engineering dalam satu fungsi.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Input data. Pandas DataFrame akan otomatis dikonversi.
    target : str, optional
        Nama kolom target untuk target encoding dan smart detection.
    strategy : str
        "smart" (default), "full", atau "minimal".
        - "smart": otomatis aktifkan fitur yang relevan berdasarkan tipe data.
        - "full" : aktifkan semua fitur.
        - "minimal": hanya transformasi dasar.
    auto_bin : bool
        Aktifkan auto binning untuk kolom numerik.
    cyclical : bool
        Aktifkan cyclical encoding untuk kolom datetime/time.
    interactions : bool
        Aktifkan pembuatan interaction terms antar kolom numerik.
    target_encoding : bool
        Aktifkan target encoding untuk kolom kategorik.
        Membutuhkan parameter `target`.
    return_pandas : bool
        Jika True, output akan dikonversi ke pandas DataFrame.
        Default False — output tetap Polars.
    config : dict, optional
        Konfigurasi granular per fitur. Contoh:
        {
            "binning": {"max_bins": 10, "method": "quantile"},
            "cyclical": ["hour", "dayofweek"],
            "interactions": ["age*income"],
        }

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        DataFrame dengan fitur tambahan. Kolom asli tetap ada.

    Examples
    --------
    >>> import polars as pl
    >>> from polarsmith import forge
    >>> df = pl.read_parquet("data/train.parquet")
    >>> df_out = forge(df, target="churn", auto_bin=True, cyclical=True)
    """
    # --- 1. Validasi dan normalisasi input ---
    df = _ensure_polars(df)
    config = config or {}
    from polarsmith._config import validate_and_normalize_config
    config = validate_and_normalize_config(config)
    _validate_args(strategy, target, target_encoding)

    # --- 2. Apply strategy override ---
    if strategy == "full":
        auto_bin = cyclical = interactions = target_encoding = True
    elif strategy == "minimal":
        auto_bin = cyclical = interactions = target_encoding = False

    # --- 3. Dispatch ke modul (import lazy untuk performa) ---
    if auto_bin:
        from polarsmith._binning import bin_features
        df = bin_features(df, config.get("binning", {}))

    if cyclical:
        from polarsmith._cyclical import add_cyclical_features
        df = add_cyclical_features(df, config.get("cyclical", None))

    if interactions:
        from polarsmith._interactions import add_interactions
        df = add_interactions(df, config.get("interactions", None))

    if target_encoding:
        if target is None:
            warnings.warn(
                "target_encoding=True tapi `target` tidak disediakan. "
                "Target encoding di-skip.",
                UserWarning,
                stacklevel=2,
            )
        else:
            from polarsmith._encoding import add_target_encoding
            df = add_target_encoding(df, target, config.get("encoding", {}))

    # --- 4. Konversi output jika diminta ---
    if return_pandas:
        return df.to_pandas()

    return df


def _ensure_polars(df: Any) -> pl.DataFrame:
    """Konversi Pandas DataFrame ke Polars jika perlu."""
    if isinstance(df, pl.DataFrame):
        return df
    try:
        return pl.from_pandas(df)
    except Exception as e:
        raise TypeError(
            f"Input harus berupa pl.DataFrame atau pd.DataFrame, "
            f"dapat: {type(df).__name__}. Error: {e}"
        ) from e


def _validate_args(strategy: str, target: str | None, target_encoding: bool) -> None:
    """Validasi argumen awal sebelum proses dimulai."""
    valid_strategies = {"smart", "full", "minimal"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy harus salah satu dari {valid_strategies}, dapat: '{strategy}'"
        )