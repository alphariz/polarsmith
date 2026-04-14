import polars as pl


def add_interactions(df: pl.DataFrame, config: dict | None = None) -> pl.DataFrame:
    """Add interactions placeholder."""
    return df.with_columns(pl.lit(1).alias("interactions_placeholder"))


