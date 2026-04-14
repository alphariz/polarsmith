import polars as pl


def add_cyclical_features(df: pl.DataFrame, config: dict | None = None) -> pl.DataFrame:
    """Add cyclical features placeholder."""
    return df.with_columns(pl.lit(1).alias("cyclical_placeholder"))


