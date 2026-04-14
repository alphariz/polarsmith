import pytest
from polarsmith._config import validate_and_normalize_config


def test_empty_config_passes():
    result = validate_and_normalize_config({})
    assert result == {}


def test_valid_binning_config():
    cfg = {"binning": {"max_bins": 5, "method": "quantile"}}
    result = validate_and_normalize_config(cfg)
    assert result["binning"]["max_bins"] == 5


def test_invalid_binning_method_raises():
    with pytest.raises(ValueError, match="harus salah satu dari"):
        validate_and_normalize_config({"binning": {"method": "invalid"}})


def test_binning_max_bins_below_2_raises():
    with pytest.raises(ValueError, match="max_bins.*harus"):
        validate_and_normalize_config({"binning": {"max_bins": 1}})



def test_valid_cyclical_config():
    cfg = {"cyclical": ["hour", "month"]}
    result = validate_and_normalize_config(cfg)
    assert result["cyclical"] == ["hour", "month"]


def test_invalid_cyclical_component_raises():
    with pytest.raises(ValueError, match="mengandung komponen tidak valid"):
        validate_and_normalize_config({"cyclical": ["hour", "zodiac"]})


def test_interactions_list_valid():
    cfg = {"interactions": ["age*income"]}
    result = validate_and_normalize_config(cfg)
    assert result["interactions"] == ["age*income"]


def test_interactions_list_bad_format_raises():
    with pytest.raises(ValueError, match=r"col_a\*col_b"):
        validate_and_normalize_config({"interactions": ["age-income"]})


def test_encoding_valid_method():
    cfg = {"encoding": {"method": "woe"}}
    result = validate_and_normalize_config(cfg)
    assert result["encoding"]["method"] == "woe"


def test_encoding_invalid_method_raises():
    with pytest.raises(ValueError, match="harus salah satu dari"):
        validate_and_normalize_config({"encoding": {"method": "mean"}})