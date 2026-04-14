import polars as pl


def bin_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Bin features placeholder."""
    return df.with_columns(pl.lit(1).alias("binning_placeholder"))


