import polars as pl


def add_target_encoding(df: pl.DataFrame, target: str, config: dict) -> pl.DataFrame:
    """Add target encoding placeholder."""
    return df.with_columns(pl.lit(1).alias("encoding_placeholder"))


