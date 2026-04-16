## Installation

```bash
pip install polarsmith
# atau dengan uv:
uv add polarsmith
```

## Quick Start

```python
import polars as pl
from polarsmith import forge

df = pl.DataFrame({
    "age": [25, 30, 35, 40],
    "income": [50000, 60000, 70000, 80000],
    "churn": [0, 1, 0, 1]
})

df_out = forge(
    df,
    target="churn",
    strategy="smart",          # auto-detect fitur yang relevan
    config={
        "binning":      {"max_bins": 10},
        "interactions": ["age*income"],
        "encoding":     {"method": "james_stein"},
    }
)
```

## Feature Matrix

| Fitur              | Status |
|--------------------|--------|
| Auto Binning       | ✅ v0.1 |
| Cyclical Features  | ✅ v0.1 |
| Interaction Terms  | ✅ v0.1 |
| James-Stein Enc.   | ✅ v0.1 |
| WoE Encoding       | ✅ v0.1 |
| Smart Detection    | ✅ v0.1 |
| Pandas Compat      | ✅ v0.1 |