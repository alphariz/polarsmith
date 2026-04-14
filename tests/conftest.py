import polars as pl
import pytest


@pytest.fixture
def df_numeric() -> pl.DataFrame:
    """DataFrame numerik murni, 500 baris."""
    import random
    random.seed(42)
    return pl.DataFrame({
        "age":    [random.randint(18, 70) for _ in range(500)],
        "income": [random.uniform(20_000, 200_000) for _ in range(500)],
        "tenure": [random.randint(0, 20) for _ in range(500)],
        "score":  [random.uniform(300, 850) for _ in range(500)],
    })


@pytest.fixture
def df_with_datetime() -> pl.DataFrame:
    """DataFrame dengan kolom datetime untuk cyclical features."""
    import random
    from datetime import datetime, timedelta
    random.seed(42)
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(hours=random.randint(0, 8760)) for _ in range(500)]
    return pl.DataFrame({
        "timestamp": pl.Series(dates).cast(pl.Datetime),
        "value":     [random.uniform(0, 100) for _ in range(500)],
    })


@pytest.fixture
def df_with_target() -> pl.DataFrame:
    """DataFrame untuk test target encoding, ada kolom binary target."""
    import random
    random.seed(42)
    categories = ["A", "B", "C", "D"]
    return pl.DataFrame({
        "category": [random.choice(categories) for _ in range(500)],
        "region":   [random.choice(["North", "South", "East", "West"]) for _ in range(500)],
        "age":      [random.randint(18, 70) for _ in range(500)],
        "churn":    [random.randint(0, 1) for _ in range(500)],
    })