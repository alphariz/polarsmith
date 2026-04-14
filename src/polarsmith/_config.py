"""
_config.py — Validasi dan normalisasi konfigurasi forge().

Memisahkan logika validasi dari _forge.py agar mudah di-test sendiri.
"""
from __future__ import annotations

from typing import Any


_VALID_BINNING_METHODS = {"equal_width", "equal_freq", "quantile"}
_VALID_ENCODING_METHODS = {"james_stein", "woe"}
_VALID_CYCLICAL_COMPONENTS = {
    "hour", "minute", "second", "dayofweek",
    "dayofmonth", "dayofyear", "month", "weekofyear",
}


def validate_and_normalize_config(config: dict) -> dict:
    """
    Validasi seluruh config dict dari user.
    Raise ValueError dengan pesan jelas jika ada yang salah.
    Return config yang sudah dinormalisasi (dengan default diisi).
    """
    config = config.copy()

    if "binning" in config:
        config["binning"] = _validate_binning_config(config["binning"])

    if "cyclical" in config:
        config["cyclical"] = _validate_cyclical_config(config["cyclical"])

    if "interactions" in config:
        config["interactions"] = _validate_interactions_config(config["interactions"])

    if "encoding" in config:
        config["encoding"] = _validate_encoding_config(config["encoding"])

    return config


def _validate_binning_config(cfg: Any) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config['binning'] harus berupa dict, dapat: {type(cfg).__name__}. "
            f"Contoh: {{'max_bins': 10, 'method': 'quantile'}}"
        )
    if "method" in cfg and cfg["method"] not in _VALID_BINNING_METHODS:
        raise ValueError(
            f"config['binning']['method'] harus salah satu dari "
            f"{_VALID_BINNING_METHODS}, dapat: '{cfg['method']}'"
        )
    if "max_bins" in cfg:
        if not isinstance(cfg["max_bins"], int) or cfg["max_bins"] < 2:
            raise ValueError(
                f"config['binning']['max_bins'] harus integer >= 2, "
                f"dapat: {cfg['max_bins']}"
            )
    return cfg


def _validate_cyclical_config(cfg: Any) -> list[str] | None:
    if cfg is None:
        return None
    if not isinstance(cfg, list):
        raise ValueError(
            f"config['cyclical'] harus list of str atau None, "
            f"dapat: {type(cfg).__name__}. "
            f"Contoh: ['hour', 'dayofweek']"
        )
    invalid = set(cfg) - _VALID_CYCLICAL_COMPONENTS
    if invalid:
        raise ValueError(
            f"config['cyclical'] mengandung komponen tidak valid: {invalid}. "
            f"Pilihan valid: {_VALID_CYCLICAL_COMPONENTS}"
        )
    return cfg


def _validate_interactions_config(cfg: Any) -> list[str] | dict:
    if isinstance(cfg, list):
        for item in cfg:
            if not isinstance(item, str) or "*" not in item:
                raise ValueError(
                    f"config['interactions'] item harus format 'col_a*col_b', "
                    f"dapat: '{item}'"
                )
        return cfg
    if isinstance(cfg, dict):
        if "max_pairs" in cfg:
            if not isinstance(cfg["max_pairs"], int) or cfg["max_pairs"] < 1:
                raise ValueError(
                    f"config['interactions']['max_pairs'] harus int >= 1, "
                    f"dapat: {cfg['max_pairs']}"
                )
        return cfg
    raise ValueError(
        f"config['interactions'] harus list atau dict, dapat: {type(cfg).__name__}"
    )


def _validate_encoding_config(cfg: Any) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config['encoding'] harus dict, dapat: {type(cfg).__name__}"
        )
    if "method" in cfg and cfg["method"] not in _VALID_ENCODING_METHODS:
        raise ValueError(
            f"config['encoding']['method'] harus salah satu dari "
            f"{_VALID_ENCODING_METHODS}, dapat: '{cfg['method']}'"
        )
    return cfg