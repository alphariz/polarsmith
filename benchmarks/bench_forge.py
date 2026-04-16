"""
Benchmark Polarsmith forge() vs pipeline Pandas equivalen.
Jalankan: python benchmarks/bench_forge.py
"""
import time
import random
import polars as pl
import pandas as pd


def make_dataset(n_rows: int) -> pl.DataFrame:
    random.seed(42)
    from datetime import datetime, timedelta
    base = datetime(2020, 1, 1)
    return pl.DataFrame({
        "age":       [random.randint(18, 70) for _ in range(n_rows)],
        "income":    [random.uniform(20_000, 200_000) for _ in range(n_rows)],
        "tenure":    [random.randint(0, 20) for _ in range(n_rows)],
        "score":     [random.uniform(300, 850) for _ in range(n_rows)],
        "timestamp": pl.Series([
            base + timedelta(hours=random.randint(0, 8760))
            for _ in range(n_rows)
        ]).cast(pl.Datetime),
    })


def benchmark_polarsmith(df: pl.DataFrame) -> float:
    from polarsmith import forge
    start = time.perf_counter()
    forge(df, auto_bin=True, cyclical=True, interactions=True,
          config={"interactions": {"max_pairs": 6}})
    return time.perf_counter() - start


def benchmark_pandas_equivalent(df_pd: pd.DataFrame) -> float:
    import numpy as np
    start = time.perf_counter()

    result = df_pd.copy()

    # Binning
    for col in ["age", "income", "tenure", "score"]:
        result[f"{col}_bin"] = pd.cut(result[col], bins=10).astype(str)

    # Cyclical
    result["hour"] = pd.to_datetime(result["timestamp"]).dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

    # Interactions (6 pairs)
    import itertools
    num_cols = ["age", "income", "tenure", "score"]
    for a, b in list(itertools.combinations(num_cols, 2))[:6]:
        result[f"{a}_x_{b}"] = result[a] * result[b]

    return time.perf_counter() - start


def run(n_rows: int) -> None:
    print(f"\n{'='*50}")
    print(f"Dataset: {n_rows:,} rows")
    print(f"{'='*50}")

    df_pl = make_dataset(n_rows)
    df_pd = df_pl.to_pandas()

    # Warmup
    benchmark_polarsmith(df_pl)
    benchmark_pandas_equivalent(df_pd)

    # Actual benchmark (3 runs, ambil minimum)
    ps_times = [benchmark_polarsmith(df_pl) for _ in range(3)]
    pd_times = [benchmark_pandas_equivalent(df_pd) for _ in range(3)]

    ps_best = min(ps_times)
    pd_best = min(pd_times)
    speedup = pd_best / ps_best

    print(f"Polarsmith : {ps_best:.3f}s")
    print(f"Pandas     : {pd_best:.3f}s")
    print(f"Speedup    : {speedup:.1f}x {'🚀' if speedup > 1 else '⚠️'}")


if __name__ == "__main__":
    for n in [10_000, 100_000, 1_000_000]:
        run(n)